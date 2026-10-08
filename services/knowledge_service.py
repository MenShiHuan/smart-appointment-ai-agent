# services/knowledge_service.py

import asyncio
import math
import threading
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from db.db_router import DatabaseRouter
from .text_embedding import embed_input


async def _embed(text: str) -> list:
    """嵌入计算放入线程池，避免同步 HTTP 调用阻塞事件循环"""
    return await asyncio.to_thread(embed_input, text)
from config import knowledge_config as kc
import logging

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "sqlite:///data/smart_appointment.db"


def _normalize(vector: List[float]) -> List[float]:
    """向量归一化：IndexFlatIP 内积在归一化后等价于余弦相似度（D1）"""
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


# ============ 进程级单例（D1） ============
# API 与咨询链路共享同一实例与索引：API 增删改后索引自动重建，
# 修复"API 新增知识对在线咨询不生效直到重启"与"每请求全量重建索引"。
_instance = None
_instance_lock = threading.Lock()


def get_knowledge_service(db_path: Optional[str] = None) -> "KnowledgeService":
    """获取进程级知识库服务单例（懒加载，线程安全；db_path 仅供测试注入，
    None 时由 DatabaseRouter 解析 DATABASE_URL 环境变量）"""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = KnowledgeService(db_path)
    return _instance


class KnowledgeService:
    """知识库服务类 - 结合数据库存储和向量检索"""

    def __init__(self, db_path: str = None):
        # 使用统一的DatabaseRouter，符合架构设计
        self.db_router = DatabaseRouter(db_path)
        self.db = self.db_router.knowledge  # 通过router访问knowledge repository
        self.index = None
        self.document_ids = []  # 维护文档ID与索引位置的映射
        self.initialized = False
        # 初始化并发锁：多调用方同时 initialize 时只构建一次索引
        self._init_lock = asyncio.Lock()
        
        # 默认知识库内容（健康理疗领域种子知识，正式内容以运营配置为准）
        self.default_knowledge = [
            {
                "content": "本健康理疗中心的营业时间是每天上午9点到晚上10点，全年无休。",
                "category": "营业时间",
                "keywords": ["营业时间", "开门", "关门", "几点", "时间"]
            },
            {
                "content": "我们提供多种理疗服务：肩颈理疗（128元/60分钟）、腰部理疗（138元/60分钟）、全身放松理疗（198元/90分钟）、足部理疗（108元/45分钟）、艾灸（98元/45分钟）。所有服务项目均明码标价、统一收费标准，价目表详见菜单，具体价格可咨询健康顾问。",
                "category": "服务项目",
                "keywords": ["服务", "理疗", "价格", "收费", "多少钱", "项目", "明码标价", "价目表"]
            },
            {
                "content": "我们有专业的男女理疗师为您服务。所有理疗师均经过系统培训，持有相关从业资格证书。您可以根据个人喜好选择男理疗师或女理疗师。",
                "category": "理疗师信息",
                "keywords": ["理疗师", "男", "女", "专业", "资格", "培训"]
            },
            {
                "content": "本中心位于市区健康生活广场2楼，交通便利。出行可在地图软件中搜索导航查询公交地铁线路，具体地址与到店指引也可致电门店咨询。",
                "category": "门店地址",
                "keywords": ["地址", "门店", "到达", "交通", "位置", "地图", "导航", "地铁"]
            },
            {
                "content": "全身放松理疗由专业理疗师采用专业手法舒缓全身，能够舒缓全身肌肉疲劳，促进血液循环，缓解压力。特别适合久坐办公室的上班族和体力劳动者。",
                "category": "服务介绍",
                "keywords": ["全身放松理疗", "效果", "作用", "好处", "适合", "疲劳", "专业手法"]
            },
            {
                "content": "肩颈理疗专门针对颈椎和肩部问题，能有效缓解颈椎疼痛、肩膀僵硬等不适。特别推荐给长期使用电脑的人群。",
                "category": "服务介绍",
                "keywords": ["肩颈理疗", "颈椎", "肩膀", "疼痛", "僵硬"]
            },
            {
                "content": "足部理疗通过刺激足部反射区，能够帮助调节全身气血运行，缓解疲劳，改善睡眠质量。",
                "category": "服务介绍",
                "keywords": ["足部理疗", "脚", "穴位", "睡眠", "疲劳"]
            },
            {
                "content": "本中心的理疗师均有3年以上专业经验，定期接受培训以确保服务质量。我们注重客户体验，力求为每位客户提供舒适的理疗服务。",
                "category": "服务质量",
                "keywords": ["经验", "专业", "培训", "质量", "舒适"]
            },
            {
                "content": "如需取消或更改预约，请提前至少2小时通知我们。临时取消可能会产生一定的费用。为保证档期，建议提前1-2天预约，热门时段请尽早安排。",
                "category": "预约政策",
                "keywords": ["取消", "更改", "改期", "退约", "政策", "提前预约"]
            },
            {
                "content": "理疗注意事项：患有高血压、心脏病、孕期、皮肤破损或近期术后的人群，请在预约前告知健康顾问，经评估确认适合后再安排理疗。",
                "category": "注意事项",
                "keywords": ["禁忌", "高血压", "孕期", "注意", "安全", "术后"]
            },
            {
                "content": "理疗结束后请适当休息并注意保暖，多饮温水，数小时内避免直接冲冷水澡与剧烈运动，避免受凉。理疗后护理对效果保持很重要。",
                "category": "注意事项",
                "keywords": ["理疗后", "保暖", "休息", "护理", "饮水", "洗澡", "冷水澡"]
            },
            {
                "content": "艾灸前请避免过饱或空腹，保持身体放松；艾灸后适当休息并注意保暖。孕期、高血压等人群请在预约前告知健康顾问，经评估确认适合后再安排。",
                "category": "注意事项",
                "keywords": ["艾灸", "过饱", "空腹", "艾灸前", "注意"]
            },
            {
                "content": "新客户首次到店可享受新人体验优惠，具体以门店当期活动为准，详情请咨询健康顾问。",
                "category": "会员服务",
                "keywords": ["新人", "首次", "优惠", "新客户", "活动"]
            },
            {
                "content": "我们提供会员卡服务，充值500元送50元，充值1000元送150元。会员还可享受预约优先权和生日优惠。",
                "category": "会员服务",
                "keywords": ["会员", "充值", "优惠", "折扣", "生日"]
            },
            {
                "content": "本店所有服务项目均明码标价、统一收费标准，价目表详见菜单，具体价格与当期活动可咨询健康顾问。",
                "category": "服务项目",
                "keywords": ["明码标价", "收费标准", "价目表", "菜单", "统一收费"]
            },
            {
                "content": "会员可享受专属折扣优惠与积分兑换服务：充值时按档位赠送余额（充值500元送50元，充值1000元送150元），并可享预约优先权与生日优惠。",
                "category": "会员服务",
                "keywords": ["会员", "专属折扣", "积分兑换", "积分", "福利"]
            },
            {
                "content": "肩颈理疗流程：理疗师先评估您的颈肩状况，通过专业手法放松颈肩部肌肉、缓解颈肩肌肉疲劳，并配合穴位按压促进血液循环，全程约60分钟。",
                "category": "服务介绍",
                "keywords": ["肩颈理疗", "流程", "做法", "怎么做", "手法", "环节"]
            },
            {
                "content": "足部理疗环节包括：足部清洁与放松、足底反射区按压、足部经络疏通与舒缓收尾，帮助调节气血运行、促进血液循环、缓解疲劳、改善睡眠，全程约45分钟。",
                "category": "服务介绍",
                "keywords": ["足部理疗", "环节", "包含", "流程", "足底", "反射区", "血液循环"]
            },
            {
                "content": "腰部理疗主要针对腰部问题：可放松腰部肌肉群、缓解腰肌劳损不适，改善久坐劳累引起的腰部肌肉紧张、酸痛。全程约60分钟。",
                "category": "服务介绍",
                "keywords": ["腰部理疗", "腰肌劳损", "腰部肌肉", "放松", "久坐", "酸痛"]
            }
        ]

    async def initialize(self):
        """初始化知识库服务（并发安全：多调用方同时初始化只构建一次）"""
        if self.initialized:
            return
        async with self._init_lock:
            if self.initialized:
                return
            try:
                # 检查数据库中是否已有数据
                existing_docs = self.db.get_all_documents()

                if not existing_docs:
                    logger.info("数据库为空，初始化默认知识库")
                    await self._create_default_knowledge()
                else:
                    logger.info(f"从数据库加载了 {len(existing_docs)} 条知识")

                # 构建向量索引
                await self._build_vector_index()
                self.initialized = True
                logger.info("知识库服务初始化完成")

            except Exception as e:
                logger.error(f"知识库服务初始化失败: {e}")
                raise

    async def _create_default_knowledge(self):
        """创建默认知识库"""
        for knowledge in self.default_knowledge:
            try:
                # 生成嵌入向量
                text_for_embedding = f"{knowledge['content']} {' '.join(knowledge['keywords'])}"
                embedding = await _embed(text_for_embedding)
                
                # 保存到数据库
                self.db.add_document(
                    content=knowledge['content'],
                    category=knowledge['category'],
                    keywords=knowledge['keywords'],
                    embedding=embedding
                )
                logger.debug(f"添加默认知识: {knowledge['content'][:50]}...")
                
            except Exception as e:
                logger.error(f"添加默认知识失败: {e}")

    async def _build_vector_index(self):
        """构建向量索引"""
        try:
            documents = self.db.get_all_documents()
            if not documents:
                logger.warning("没有文档可用于构建索引")
                return

            embeddings = []
            self.document_ids = []
            
            for doc in documents:
                if doc.get('embedding'):
                    embeddings.append(doc['embedding'])
                    self.document_ids.append(doc['id'])
                else:
                    # 如果没有嵌入向量，生成一个
                    logger.warning(f"文档 {doc['id']} 缺少嵌入向量，正在生成...")
                    text_for_embedding = f"{doc['content']} {' '.join(doc.get('keywords', []))}"
                    embedding = await _embed(text_for_embedding)
                    
                    # 更新数据库
                    self.db.update_document(doc['id'], embedding=embedding)
                    
                    embeddings.append(embedding)
                    self.document_ids.append(doc['id'])

            if embeddings:
                # 创建FAISS索引（入库向量归一化：内积等价余弦相似度，D1）
                embeddings_array = np.array(
                    [_normalize(emb) for emb in embeddings]).astype('float32')
                dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
                self.index.add(embeddings_array)
                logger.info(f"构建向量索引完成，包含 {len(embeddings)} 个向量")
            else:
                logger.warning("没有有效的嵌入向量，无法构建索引")

        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            raise

    async def search(self, query: str, top_k: int = 3, category: str = None) -> List[Dict]:
        """搜索相关文档（D1：归一化余弦分数 + 批量回查 + 相似度下限过滤）"""
        if not self.initialized or self.index is None or not self.document_ids:
            logger.warning("知识库服务未初始化或索引不可用")
            return []

        try:
            # 生成查询的嵌入向量（归一化，与入库口径一致）
            query_embedding = _normalize(await _embed(query))
            query_array = np.array([query_embedding]).astype('float32')

            # 向量搜索（多检索一些候选，为分类过滤留余量）
            candidate_k = min(top_k * kc.CANDIDATE_MULTIPLIER, len(self.document_ids))
            scores, indices = self.index.search(query_array, candidate_k)

            # 批量回查文档（一次 IN 查询，软删除文档自动过滤）
            doc_ids = [self.document_ids[idx] for idx in indices[0]
                       if 0 <= idx < len(self.document_ids)]
            docs_by_id = self.db.get_documents_by_ids(doc_ids)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.document_ids):
                    continue
                score = float(score)
                if score < kc.MIN_SIMILARITY:
                    continue  # 相似度下限：无关候选不注入
                doc_id = self.document_ids[idx]
                doc = docs_by_id.get(doc_id)
                if not doc:
                    continue
                if category and doc.get('category') != category:
                    continue

                doc = dict(doc)
                doc['score'] = score
                doc['rank'] = len(results) + 1
                results.append(doc)

                if len(results) >= top_k:
                    break

            return results

        except Exception as e:
            logger.error(f"搜索知识库失败: {e}")
            return []

    async def add_document(self, content: str, category: str, keywords: List[str] = None) -> bool:
        """添加新文档"""
        try:
            if keywords is None:
                keywords = []
            
            # 生成嵌入向量
            text_for_embedding = f"{content} {' '.join(keywords)}"
            embedding = await _embed(text_for_embedding)
            
            # 保存到数据库
            doc_id = self.db.add_document(content, category, keywords, embedding)
            
            # 重建索引
            await self._build_vector_index()
            
            logger.info(f"成功添加文档 {doc_id}: {content[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    async def update_document(self, doc_id: int, content: str = None, category: str = None, keywords: List[str] = None) -> bool:
        """更新文档"""
        try:
            # 如果更新了内容或关键词，需要重新生成嵌入向量
            embedding = None
            if content is not None or keywords is not None:
                # 获取当前文档信息
                current_doc = self.db.get_document(doc_id)
                if not current_doc:
                    return False
                
                # 使用新值或保持原值
                final_content = content if content is not None else current_doc['content']
                final_keywords = keywords if keywords is not None else current_doc.get('keywords', [])
                
                # 生成新的嵌入向量
                text_for_embedding = f"{final_content} {' '.join(final_keywords)}"
                embedding = await _embed(text_for_embedding)
            
            # 更新数据库
            success = self.db.update_document(doc_id, content, category, keywords, embedding)
            
            if success and embedding is not None:
                # 重建索引
                await self._build_vector_index()
            
            return success
            
        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return False

    async def delete_document(self, doc_id: int, soft_delete: bool = True) -> bool:
        """删除文档"""
        try:
            success = self.db.delete_document(doc_id, soft_delete)
            
            if success:
                # 重建索引
                await self._build_vector_index()
            
            return success
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False

    def get_all_documents(self, include_inactive: bool = False) -> List[Dict]:
        """获取所有文档"""
        return self.db.get_all_documents(include_inactive)

    def get_document(self, doc_id: int) -> Dict:
        """获取指定文档"""
        return self.db.get_document(doc_id)

    def get_all_categories(self) -> List[str]:
        """获取所有分类"""
        return self.db.get_all_categories()

    def get_documents_count(self) -> int:
        """获取文档总数"""
        return self.db.get_documents_count()

    def search_by_category(self, category: str) -> List[Dict]:
        """按分类搜索文档"""
        return self.db.search_documents_by_category(category)

    def search_by_keywords(self, keywords: List[str]) -> List[Dict]:
        """按关键词搜索文档"""
        return self.db.search_documents_by_keywords(keywords)
