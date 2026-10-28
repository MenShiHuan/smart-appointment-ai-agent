"""
向量检索与入库管道配置（RAG 重构：Milvus 双 Function 混合检索）

按模块分组：
- Milvus 连接与集合（BM25 稀疏 + DashScope 稠密 + 语义精排 Function）
- DashScope 云 API（编码/精排以 Milvus 服务端 Function 调用为主，应用侧仅兜底直连）
- 检索流水线参数（两路召回 → RRF 融合 → 语义精排）
- 文本切分参数（结构感知 + 递归字符 + 滑动窗口）
- MinerU 文档解析服务

全部参数可用环境变量覆盖。
"""
from config.env import _env_str, _env_int, _env_float


# ============ Milvus 连接与集合 ============
MILVUS_URI = _env_str("MILVUS_URI", "http://localhost:19530")
# 无鉴权部署时为空串（standalone 默认无需 token）
MILVUS_TOKEN = _env_str("MILVUS_TOKEN", "")
MILVUS_COLLECTION = _env_str("MILVUS_COLLECTION", "knowledge_chunks")

# 稠密向量维度（text-embedding-v3 默认 1024，与字段定义一致）
DENSE_DIM = _env_int("MILVUS_DENSE_DIM", 1024)

# content 字段 VARCHAR 上限保护（Milvus 字段 max_length 需 >= 实际文本长度）
CONTENT_MAX_CHARS = _env_int("CHUNK_CONTENT_MAX_CHARS", 2000)
ID_MAX_LEN = 128          # chunk_id = {doc_id}_{seq}（64 hex + _ + seq）
DOC_ID_MAX_LEN = 64       # SHA-256 完整 hex / text_<sqlite_id>
CHAPTER_TITLE_MAX_LEN = 256

# ============ DashScope 云 API（全链路向量化 text-embedding-v3 + 精排 gte-rerank-v2） ============
# DASHSCOPE_API_KEY：向量化凭证（应用侧 text_embedding / 检索兜底编码）
# DASHSCOPE_RERANK_API_KEY：精排凭证（应用侧兜底直连；未配置时回退 DASHSCOPE_API_KEY）
# Milvus Function 路径由 Milvus 服务端出网调用，其容器侧配置同名环境变量
# （DASHSCOPE_API_KEY / DASHSCOPE_RERANK_API_KEY，勿加 MILVUSAI 前缀）。
DASHSCOPE_API_KEY = _env_str("DASHSCOPE_API_KEY", "")
DASHSCOPE_RERANK_API_KEY = _env_str("DASHSCOPE_RERANK_API_KEY", "")
EMBED_PROVIDER = _env_str("MILVUS_EMBED_PROVIDER", "dashscope")
EMBED_MODEL = _env_str("MILVUS_EMBED_MODEL", "text-embedding-v3")
RERANK_MODEL = _env_str("MILVUS_RERANK_MODEL", "gte-rerank-v2")
# 精排后端：milvus（内嵌 RERANK Function，默认）/ dashscope（应用侧直连兜底）/ none（跳过精排）
RERANK_BACKEND = _env_str("RERANK_BACKEND", "milvus").lower()
DASHSCOPE_EMBED_URL = _env_str(
    "DASHSCOPE_EMBED_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
)
DASHSCOPE_RERANK_URL = _env_str(
    "DASHSCOPE_RERANK_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
)

# ============ 检索流水线参数 ============
# 单路召回放大倍数：最终要 top_k 条时，每路多召回候选保证融合后充足
CANDIDATE_MULTIPLIER = _env_int("HYBRID_CANDIDATE_MULTIPLIER", 2)
# 单路召回下限（稠密/稀疏各一路）
PER_PATH_TOP_K = _env_int("HYBRID_PER_PATH_TOP_K", 10)
# RRF 融合参数 k（Milvus 推荐区间 [10, 100]）
RRF_K = _env_int("RRF_K", 60)
# RRF 融合后保留候选数（送入精排）
FUSION_TOP_K = _env_int("FUSION_TOP_K", 5)
# 精排分数下限：None 表示不过滤；精排未启用/降级时该参数不生效
RERANK_MIN_SCORE = _env_float("RERANK_MIN_SCORE", 0.0) or None

# ============ 文本切分参数 ============
# 递归字符切分目标长度（token）与滑动窗口重叠（token）
CHUNK_SIZE_TOKENS = _env_int("CHUNK_SIZE_TOKENS", 512)
CHUNK_OVERLAP_TOKENS = _env_int("CHUNK_OVERLAP_TOKENS", 128)
# token 估算：中文 1 字 ≈ 1 token；英文场景可下调（如 4 字符 ≈ 1 token 则配 4.0）
CHARS_PER_TOKEN = _env_float("CHARS_PER_TOKEN", 1.0)

# ============ MinerU 云文档解析服务（官方 OpenAPI v4，本地不部署） ============
MINERU_BASE_URL = _env_str("MINERU_BASE_URL", "https://mineru.net/api/v4")
# 云 API 凭证（.env 的 MINERU_TOKEN，Authorization: Bearer）
MINERU_TOKEN = _env_str("MINERU_TOKEN", "")
# 解析模式：vlm=精准解析（默认）/ pipeline=普通
MINERU_MODEL_VERSION = _env_str("MINERU_MODEL_VERSION", "vlm")
# 端点路径（v4 三端点：申请预签名上传地址 / 触发解析 / 查询结果）
MINERU_UPLOAD_PATH = _env_str("MINERU_UPLOAD_PATH", "/file-urls/batch")
MINERU_PARSE_PATH = _env_str("MINERU_PARSE_PATH", "/extract/task/batch")
MINERU_RESULT_PATH = _env_str("MINERU_RESULT_PATH", "/extract-results/batch/{batch_id}")
# 单个文档解析最长等待（秒）；解析失败走 ETL 状态机记录
MINERU_MAX_WAIT = _env_int("MINERU_MAX_WAIT", 600)
# 任务轮询间隔（秒）
MINERU_POLL_INTERVAL = _env_float("MINERU_POLL_INTERVAL", 2.0)
# 单次 HTTP 超时（秒）
MINERU_TIMEOUT = _env_int("MINERU_TIMEOUT", 300)
