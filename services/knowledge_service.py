"""
知识库服务（RAG 重构）：SQLite 元数据 registry + Milvus 双向量混合检索

- SQLite（KnowledgeDocument 表）：文本知识的元数据/内容 registry，web CRUD 数据源
- Milvus：唯一向量库，单 Collection 承载 BM25 稀疏（内嵌 Function）+
  DashScope 稠密（TEXTEMBEDDING Function）+ 语义精排（RERANK Function）

对外契约与旧 FAISS 实现保持一致（方法签名/返回结构），
API 层、咨询链路、评测链路无感切换。进程级单例共享同一 Milvus 客户端。
"""
# services/knowledge_service.py

import asyncio
import logging
import threading
from typing import List, Dict, Optional, Any

from db.db_router import DatabaseRouter
from services.vector_store import get_vector_store
from config import vector_config as vc

logger = logging.getLogger(__name__)


# ============ 进程级单例 ============
# API 与咨询链路共享同一实例与 Milvus 客户端：API 增删改后在线检索立即可见。
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
    """知识库服务类 - SQLite 存储 + Milvus 双向量混合检索"""

    def __init__(self, db_path: str = None):
        # 使用统一的DatabaseRouter，符合架构设计
        self.db_router = DatabaseRouter(db_path)
        self.db = self.db_router.knowledge  # 通过router访问knowledge repository
        self.vector_store = get_vector_store()
        self.initialized = False
        # 初始化并发锁：多调用方同时 initialize 时只构建一次
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
            }
        ]

    async def initialize(self):
        """初始化知识库服务（并发安全：多调用方同时初始化只构建一次）

        失败抛异常（app.py 快速失败语义）；成功后 initialized=True，幂等可重入。
        """
        if self.initialized:
            return
        async with self._init_lock:
            if self.initialized:
                return
            # 向量库初始化（建集合/加载，Milvus 不可用在此抛错）
            await self.vector_store.initialize()

            existing_docs = self.db.get_all_documents()
            if not existing_docs:
                logger.info("数据库为空，初始化默认知识库")
                await self._create_default_knowledge()
            else:
                logger.info(f"从数据库加载了 {len(existing_docs)} 条知识")
                # 存量升级：FAISS 时代的文档在 Milvus 中缺失，按需回填
                await self._backfill_text_docs(existing_docs)

            self.initialized = True
            logger.info("知识库服务初始化完成")

    async def _create_default_knowledge(self):
        """创建默认知识库（SQLite 注册 + Milvus 双向量自动编码）"""
        for knowledge in self.default_knowledge:
            try:
                doc_id = self.db.add_document(
                    content=knowledge['content'],
                    category=knowledge['category'],
                    keywords=knowledge['keywords'],
                )
                chunk = self._text_chunk(doc_id, knowledge['content'], knowledge['category'])
                await self.vector_store.upsert_chunks([chunk])
                logger.debug(f"添加默认知识: {knowledge['content'][:50]}...")
            except Exception as e:
                logger.error(f"添加默认知识失败: {e}")

    async def _backfill_text_docs(self, existing_docs: List[Dict]):
        """存量文本知识回填 Milvus（旧 FAISS 数据升级场景，幂等 upsert）"""
        try:
            present_ids = await self.vector_store.get_doc_ids(doc_type="text")
        except Exception as e:
            logger.error(f"查询向量库文本知识失败: {e}")
            return
        missing = [
            d for d in existing_docs
            if d.get("is_active") and f"text_{d['id']}" not in present_ids
        ]
        if not missing:
            return
        chunks = [self._text_chunk(d["id"], d["content"], d["category"]) for d in missing]
        await self.vector_store.upsert_chunks(chunks)
        logger.info(f"已回填 {len(chunks)} 条存量知识到向量库")

    def _text_chunk(self, doc_id: int, content: str, category: str) -> Dict[str, Any]:
        """文本知识条目 → Milvus chunk 字典（单条目单 chunk，seq=1）"""
        return {
            "id": f"text_{doc_id}_1",
            "doc_id": f"text_{doc_id}",
            "doc_type": "text",
            "page_num": 0,
            "chapter_title": category or "",
            "category": category or "",
            "content": content[:vc.CONTENT_MAX_CHARS],
        }

    async def search(self, query: str, top_k: int = 3, category: str = None) -> List[Dict]:
        """搜索相关文档（混合检索：BM25 稀疏 + DashScope 稠密 → RRF 融合 → 语义精排）

        未初始化/异常返回空列表；结果按契约组装（文本知识含 SQLite 八字段，
        PDF 分片含 page_num/chapter_title 等元数据），score 为精排分（降级时
        为归一化 RRF 分），rank 从 1 起。
        """
        if not self.initialized:
            logger.warning("知识库服务未初始化")
            return []

        try:
            candidates = await self.vector_store.hybrid_search(
                query, top_k=top_k, category=category
            )
            candidates = await self.vector_store.rerank(query, candidates)

            # 精排分数下限（仅对精排成功的候选生效）
            if vc.RERANK_MIN_SCORE:
                candidates = [
                    c for c in candidates
                    if c.get("score_source") != "rerank" or c["score"] >= vc.RERANK_MIN_SCORE
                ]

            return self._enrich_results(candidates[:top_k])

        except Exception as e:
            logger.error(f"搜索知识库失败: {e}")
            return []

    def _enrich_results(self, candidates: List[Dict]) -> List[Dict]:
        """Milvus 命中 → 契约字典：文本知识从 SQLite 补齐八字段，PDF 分片构造精简字典"""
        text_sql_ids = []
        for c in candidates:
            doc_id = str(c.get("doc_id") or "")
            if c.get("doc_type") == "text" and doc_id.startswith("text_"):
                try:
                    text_sql_ids.append(int(doc_id[len("text_"):]))
                except ValueError:
                    pass
        docs_by_id = self.db.get_documents_by_ids(text_sql_ids) if text_sql_ids else {}

        results = []
        rank = 0
        for c in candidates:
            if c.get("doc_type") == "text":
                doc_id = str(c.get("doc_id") or "")
                try:
                    sql_id = int(doc_id[len("text_"):])
                except ValueError:
                    continue
                repo = docs_by_id.get(sql_id)
                if not repo:
                    continue  # 软删除或已不存在
                rank += 1
                results.append({
                    **repo,
                    "score": c["score"],
                    "rank": rank,
                    "doc_id": doc_id,
                    "doc_type": "text",
                    "page_num": 0,
                    "chapter_title": c.get("chapter_title") or repo.get("category", ""),
                })
            else:
                rank += 1
                results.append({
                    "id": c["id"],
                    "content": c.get("content", ""),
                    "category": "",
                    "keywords": [],
                    "embedding": None,
                    "created_at": None,
                    "updated_at": None,
                    "is_active": True,
                    "score": c["score"],
                    "rank": rank,
                    "doc_id": c.get("doc_id"),
                    "doc_type": c.get("doc_type", "pdf"),
                    "page_num": c.get("page_num"),
                    "chapter_title": c.get("chapter_title") or "",
                })
        return results

    async def add_document(self, content: str, category: str, keywords: List[str] = None) -> bool:
        """添加新文档（SQLite 注册 + Milvus 双向量自动编码，写后在线检索立即可见）"""
        try:
            if keywords is None:
                keywords = []

            doc_id = self.db.add_document(content, category, keywords)
            chunk = self._text_chunk(doc_id, content, category)
            await self.vector_store.upsert_chunks([chunk])

            logger.info(f"成功添加文档 {doc_id}: {content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    async def update_document(self, doc_id: int, content: str = None, category: str = None, keywords: List[str] = None) -> bool:
        """更新文档（SQLite 更新 + Milvus 幂等 upsert；chunk_id 确定性不变，
        向量随 content 由 Function 重新生成）"""
        try:
            current = self.db.get_document(doc_id)
            if not current:
                return False

            success = self.db.update_document(doc_id, content, category, keywords)
            if not success:
                return False

            final_content = content if content is not None else current['content']
            final_category = category if category is not None else current['category']
            chunk = self._text_chunk(doc_id, final_content, final_category)
            await self.vector_store.upsert_chunks([chunk])

            return success

        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return False

    async def delete_document(self, doc_id: int, soft_delete: bool = True) -> bool:
        """删除文档（SQLite 软删除 + Milvus 按 doc_id 清除 chunk，即时生效）"""
        try:
            success = self.db.delete_document(doc_id, soft_delete)

            if success:
                await self.vector_store.delete_by_doc_id(f"text_{doc_id}")

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
