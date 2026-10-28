"""
Milvus 向量库封装（RAG 重构）

单 Collection 同时承载：
- 稀疏向量：Milvus 内嵌 BM25 Function（content 字段中文分词 analyzer）
- 稠密向量：Milvus 内嵌 TEXTEMBEDDING Function（DashScope text-embedding-v3，1024 维）
- 语义精排：Milvus 内嵌 RERANK Function（DashScope gte-rerank-v2）

检索链路：两路 AnnSearchRequest（data 传原始文本，由 Function 服务端编码）
→ RRFRanker(k) 融合 → 语义精排 → 返回带元数据与统一分数口径的结果。

设计要点：
- pymilvus 为同步 API，全部经 asyncio.to_thread 包装成 async 方法；
- 模块允许在未安装 pymilvus 时导入，initialize() 时才抛明确错误
  （app.py 快速失败语义承接）；
- 服务端不支持文本直搜 / RERANK 动态 query 时自动降级：
  稠密=应用侧直连 DashScope、稀疏=analyze_text、精排=dashscope 直连/透传，
  保证检索链路可用。
"""
import asyncio
import json
import logging
import threading
from typing import List, Dict, Optional, Any

from config import vector_config as vc

logger = logging.getLogger(__name__)

try:
    from pymilvus import (
        MilvusClient,
        DataType,
        Function,
        FunctionType,
        AnnSearchRequest,
        RRFRanker,
    )
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False


def _require_pymilvus():
    """pymilvus 未安装时给出明确指引（部署由用户负责，安装后即恢复）"""
    if not PYMILVUS_AVAILABLE:
        raise RuntimeError(
            "pymilvus 未安装：请先 pip install pymilvus（版本随部署的 Milvus ≥2.6 选择）"
        )


# ============ 进程级单例（与 KnowledgeService 同进程共享客户端） ============
_instance = None
_instance_lock = threading.Lock()


def get_vector_store() -> "VectorStore":
    """获取进程级向量库单例（懒加载，线程安全）"""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = VectorStore()
    return _instance


def _expr_str(value: str) -> str:
    """标量过滤表达式中的字符串转义（防注入/引号截断）"""
    return json.dumps(str(value), ensure_ascii=False)


class VectorStore:
    """Milvus 客户端封装：集合生命周期、chunk 写入/删除、混合检索、语义精排"""

    # 检索回传的标量元数据字段
    OUTPUT_FIELDS = ["doc_id", "doc_type", "page_num", "chapter_title", "category", "content"]

    def __init__(self, uri: str = None, collection_name: str = None):
        self.uri = uri or vc.MILVUS_URI
        self.collection_name = collection_name or vc.MILVUS_COLLECTION
        self.client = None
        self.initialized = False
        # 并发初始化锁：多调用方同时 initialize 时只建一次集合
        self._init_lock = asyncio.Lock()

    # ============ 连接与集合生命周期 ============

    def _connect(self):
        if self.client is None:
            _require_pymilvus()
            kwargs = {"uri": self.uri}
            if vc.MILVUS_TOKEN:
                kwargs["token"] = vc.MILVUS_TOKEN
            self.client = MilvusClient(**kwargs)

    def _ensure_collection_sync(self):
        """建集合（schema + 3 个 Function + 索引）并加载；已存在则复用并重载"""
        self._connect()
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            return

        schema = self.client.create_schema(
            auto_id=False,           # id 由应用侧生成（两级哈希 ID 策略）
            enable_dynamic_field=False,
        )
        # 主键：chunk_id = {doc_id}_{seq}（VARCHAR 主键，Milvus 2.4+ 支持）
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR,
            max_length=vc.ID_MAX_LEN, is_primary=True,
        )
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=vc.DOC_ID_MAX_LEN)
        schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=16)
        schema.add_field(field_name="page_num", datatype=DataType.INT32)
        schema.add_field(
            field_name="chapter_title", datatype=DataType.VARCHAR,
            max_length=vc.CHAPTER_TITLE_MAX_LEN,
        )
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=128)
        # 原始文本：BM25 Function 输入（中文分词）+ 稠密编码输入
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=vc.CONTENT_MAX_CHARS + 64,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=vc.DENSE_DIM)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 1. BM25 Function：content → sparse（稀疏向量由 Milvus 内嵌 BM25 生成）
        schema.add_function(Function(
            name="bm25_emb",
            input_field_names=["content"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        ))
        # 2. TEXTEMBEDDING Function：content → dense（DashScope 云 API；
        #    凭证为 Milvus 服务端容器 env DASHSCOPE_API_KEY，勿加 MILVUSAI 前缀）
        schema.add_function(Function(
            name="dashscope_emb",
            input_field_names=["content"],
            output_field_names=["dense"],
            function_type=FunctionType.TEXTEMBEDDING,
            params={
                "provider": vc.EMBED_PROVIDER,
                "model_name": vc.EMBED_MODEL,
                "dim": str(vc.DENSE_DIM),
            },
        ))
        # 3. RERANK Function：语义精排（gte-rerank-v2；凭证为 Milvus 服务端
        #    env DASHSCOPE_RERANK_API_KEY，勿加前缀）。queries 随每次检索调用动态传入
        schema.add_function(Function(
            name="gte_rerank",
            input_field_names=["content"],
            function_type=FunctionType.RERANK,
            params={
                "reranker": "model",
                "provider": "ali",
                "model_name": vc.RERANK_MODEL,
                "max_client_batch_size": 128,
            },
        ))

        # 索引：稠密 COSINE；稀疏 BM25 倒排；doc_id/category 标量过滤索引
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense", index_type="AUTOINDEX", metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
        )
        index_params.add_index(field_name="doc_id", index_type="AUTOINDEX")
        index_params.add_index(field_name="category", index_type="AUTOINDEX")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(self.collection_name)
        logger.info("已创建并加载 Milvus 集合 %s", self.collection_name)

    async def initialize(self):
        """初始化向量库（幂等；失败抛异常，由上层快速失败承接）"""
        if self.initialized:
            return
        async with self._init_lock:
            if self.initialized:
                return
            await asyncio.to_thread(self._ensure_collection_sync)
            self.initialized = True
            logger.info("向量库初始化完成（Milvus 双向量混合检索）")

    # ============ 写入与删除 ============

    async def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """批量写入 chunk（id 确定性：重复写入即覆盖更新；向量由 Function 自动生成）"""
        if not chunks:
            return True
        try:
            return await asyncio.to_thread(self._upsert_sync, chunks)
        except Exception as e:
            logger.error("写入 Milvus 失败: %s", e)
            return False

    def _upsert_sync(self, chunks: List[Dict[str, Any]]) -> bool:
        self._connect()
        rows = [
            {
                "id": c["id"],
                "doc_id": c["doc_id"],
                "doc_type": c["doc_type"],
                "page_num": c["page_num"],
                "chapter_title": c["chapter_title"],
                "category": c["category"],
                "content": c["content"],
            }
            for c in chunks
        ]
        self.client.upsert(collection_name=self.collection_name, data=rows)
        return True

    async def delete_by_doc_id(self, doc_id: str) -> bool:
        """按文档维度删除全部 chunk（文档更新语义：先删后重入库）"""
        try:
            expr = f"doc_id == {_expr_str(doc_id)}"
            await asyncio.to_thread(
                self.client.delete, collection_name=self.collection_name, filter=expr
            )
            return True
        except Exception as e:
            logger.error("按 doc_id 删除失败 (%s): %s", doc_id, e)
            return False

    async def get_row_count(self) -> int:
        """集合内 chunk 总数（种子知识回填判断用）"""
        try:
            stats = await asyncio.to_thread(
                self.client.get_collection_stats, self.collection_name
            )
            return int(stats.get("row_count", 0))
        except Exception as e:
            logger.error("获取集合统计失败: %s", e)
            return 0

    async def get_doc_ids(self, doc_type: Optional[str] = None) -> set:
        """集合内已存在的 doc_id 集合（存量文本知识回填判断用）"""
        try:
            kwargs = dict(
                collection_name=self.collection_name,
                filter=f"doc_type == {_expr_str(doc_type)}" if doc_type else "",
                output_fields=["doc_id"],
                limit=10000,
            )
            rows = await asyncio.to_thread(self.client.query, **kwargs)
            return {row.get("doc_id") for row in rows}
        except Exception as e:
            logger.error("查询 doc_id 列表失败: %s", e)
            return set()

    # ============ 混合检索 ============

    @staticmethod
    def _build_filter_expr(doc_id: Optional[str], category: Optional[str]) -> Optional[str]:
        """组装标量过滤表达式（doc_id 文档范围过滤 / category 分类过滤）"""
        parts = []
        if doc_id:
            parts.append(f"doc_id == {_expr_str(doc_id)}")
        if category:
            parts.append(f"category == {_expr_str(category)}")
        return " and ".join(parts) if parts else None

    async def hybrid_search(
        self,
        query_text: str,
        top_k: int = 3,
        doc_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """两路混合检索（稠密 + BM25 稀疏）→ RRF 融合。

        返回候选列表：{id, score(融合分), doc_id, doc_type, page_num,
        chapter_title, category, content}；失败返回空列表。
        """
        if not self.initialized:
            logger.warning("向量库未初始化，跳过检索")
            return []
        try:
            return await asyncio.to_thread(
                self._hybrid_search_sync, query_text, top_k, doc_id, category
            )
        except Exception as e:
            logger.error("混合检索失败: %s", e)
            return []

    def _hybrid_search_sync(
        self,
        query_text: str,
        top_k: int,
        doc_id: Optional[str],
        category: Optional[str],
    ) -> List[Dict[str, Any]]:
        per_path_k = max(vc.PER_PATH_TOP_K, top_k * vc.CANDIDATE_MULTIPLIER)
        fusion_limit = max(vc.FUSION_TOP_K, top_k)
        expr = self._build_filter_expr(doc_id, category)

        try:
            # 主路径：data 传原始文本，由 Function 服务端编码（稀疏=BM25 分词，稠密=DashScope）
            dense_req = AnnSearchRequest(
                data=[query_text], anns_field="dense",
                param={"metric_type": "COSINE"}, limit=per_path_k,
            )
            sparse_req = AnnSearchRequest(
                data=[query_text], anns_field="sparse",
                param={"metric_type": "BM25"}, limit=per_path_k,
            )
            kwargs = dict(
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(vc.RRF_K),
                limit=fusion_limit,
                output_fields=self.OUTPUT_FIELDS,
            )
            if expr:
                kwargs["filter"] = expr
            raw = self.client.hybrid_search(self.collection_name, **kwargs)
        except Exception as e:
            # 兜底：服务端不支持文本直搜 → 应用侧编码 query 向量后检索
            logger.warning("文本直搜失败（%s），降级为应用侧编码查询向量", e)
            raw = self._hybrid_search_with_vectors_sync(query_text, per_path_k, fusion_limit, expr)

        hits = raw[0] if raw else []
        return [self._hit_to_dict(h) for h in hits if h]

    def _hybrid_search_with_vectors_sync(
        self, query_text: str, per_path_k: int, fusion_limit: int, expr: Optional[str]
    ) -> List:
        """兜底路径：稠密向量应用侧直连 DashScope，稀疏向量 analyze_text（BM25 同口径）"""
        dense_vec = self._embed_query_dashscope_sync(query_text)
        sparse_vec = self._analyze_text_sync(query_text)
        dense_req = AnnSearchRequest(
            data=[dense_vec], anns_field="dense",
            param={"metric_type": "COSINE"}, limit=per_path_k,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec], anns_field="sparse",
            param={"metric_type": "IP"}, limit=per_path_k,  # 显式稀疏向量用 IP
        )
        kwargs = dict(
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(vc.RRF_K),
            limit=fusion_limit,
            output_fields=self.OUTPUT_FIELDS,
        )
        if expr:
            kwargs["filter"] = expr
        return self.client.hybrid_search(self.collection_name, **kwargs)

    def _embed_query_dashscope_sync(self, text: str) -> List[float]:
        """应用侧直连 DashScope 编码 query（兜底路径）"""
        import requests

        if not vc.DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置，无法应用侧编码 query")
        resp = requests.post(
            vc.DASHSCOPE_EMBED_URL,
            json={"model": vc.EMBED_MODEL, "input": text, "dimensions": vc.DENSE_DIM},
            headers={"Authorization": f"Bearer {vc.DASHSCOPE_API_KEY}"},
            timeout=60,
        )
        resp.raise_for_status()
        return [float(x) for x in resp.json()["data"][0]["embedding"]]

    def _analyze_text_sync(self, text: str):
        """服务端 BM25 同口径的 query 稀疏向量（pymilvus 2.6 analyze_text）"""
        analyze = getattr(self.client, "analyze_text", None)
        if analyze is None:
            raise RuntimeError("当前 pymilvus 版本无 analyze_text 接口")
        results = analyze(self.collection_name, [text])
        return results[0]

    @staticmethod
    def _hit_to_dict(hit: Dict) -> Dict[str, Any]:
        entity = hit.get("entity") or {}
        return {
            "id": hit.get("id"),
            "score": float(hit.get("distance", 0.0)),
            **{f: entity.get(f) for f in VectorStore.OUTPUT_FIELDS},
        }

    # ============ 语义精排 ============

    async def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对 RRF 融合候选做语义精排（gte-rerank-v2），按 score 降序返回。

        每个候选附加 score（统一口径）与 score_source：
        - "rerank"：精排相关性分（0..1），可配合 RERANK_MIN_SCORE 过滤
        - "rrf"：精排未启用/降级时，RRF 融合分归一化到 [0,1]
        任一后端失败时逐级降级（milvus → dashscope → 透传），保证链路可用。
        """
        if not candidates:
            return []
        if vc.RERANK_BACKEND == "none":
            return self._normalize_rrf_scores(candidates)

        for backend in ("milvus", "dashscope"):
            if backend != vc.RERANK_BACKEND:
                continue
            try:
                scores = await asyncio.to_thread(
                    self._rerank_scores_sync, backend, query, candidates
                )
                ranked = [
                    {**c, "score": scores.get(c["id"], c["score"]), "score_source": "rerank"}
                    for c in candidates
                ]
                ranked.sort(key=lambda c: c["score"], reverse=True)
                return ranked
            except Exception as e:
                logger.warning("精排后端 %s 失败（%s），降级", backend, e)

        logger.warning("精排全部降级，透传 RRF 融合顺序")
        return self._normalize_rrf_scores(candidates)

    def _rerank_scores_sync(
        self, backend: str, query: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        if backend == "milvus":
            return self._rerank_via_milvus_sync(query, candidates)
        return self._rerank_via_dashscope_sync(query, candidates)

    def _rerank_via_milvus_sync(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """内嵌 RERANK Function 路径：每查询动态构造 ranker（queries 随调用传入）"""
        ranker = Function(
            name="gte_rerank",
            input_field_names=["content"],
            function_type=FunctionType.RERANK,
            params={
                "reranker": "model",
                "provider": "ali",
                "model_name": vc.RERANK_MODEL,
                "queries": [query],
                "max_client_batch_size": min(len(candidates), 128),
            },
        )
        ids = [_expr_str(c["id"]) for c in candidates]
        raw = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="dense",
            limit=len(candidates),
            output_fields=self.OUTPUT_FIELDS,
            ranker=ranker,
            filter=f"id in [{', '.join(ids)}]",
        )
        hits = raw[0] if raw else []
        return {h["id"]: float(h.get("distance", 0.0)) for h in hits if h}

    def _rerank_via_dashscope_sync(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """应用侧直连 DashScope rerank API（与 Milvus Function 同模型同 provider）"""
        import requests

        api_key = vc.DASHSCOPE_RERANK_API_KEY or vc.DASHSCOPE_API_KEY
        if not api_key:
            raise RuntimeError("DASHSCOPE_RERANK_API_KEY 未配置，无法直连 DashScope 精排")
        resp = requests.post(
            vc.DASHSCOPE_RERANK_URL,
            json={
                "model": vc.RERANK_MODEL,
                "input": {"query": query, "documents": [c["content"] for c in candidates]},
                "parameters": {"return_documents": False, "top_n": len(candidates)},
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        resp.raise_for_status()
        scores: Dict[str, float] = {}
        for item in resp.json().get("output", {}).get("results", []):
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(candidates):
                scores[candidates[idx]["id"]] = float(item.get("relevance_score", 0.0))
        return scores

    @staticmethod
    def _normalize_rrf_scores(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """RRF 融合分归一化到 [0,1]（除以单文档两路都排第 1 的理论最大分）"""
        max_score = 2.0 / (vc.RRF_K + 1) or 1.0
        ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
        return [
            {**c, "score": min(float(c["score"]) / max_score, 1.0), "score_source": "rrf"}
            for c in ranked
        ]
