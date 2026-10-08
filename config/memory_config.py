"""
离线记忆整合参数配置（C4a）

所有触发阈值与权重集中在此，便于调参；设计口径见 DEV_SPEC 2.6 / 2.6.1。
"""
import os


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


# 触发条件
MIN_SESSIONS = _env_int("MEMORY_MIN_SESSIONS", 5)          # 新增会话数达到该值才触发整合
MIN_INTERVAL_HOURS = _env_int("MEMORY_MIN_INTERVAL_HOURS", 24)  # 距上次整合的最小间隔（小时）

# 回放与批处理
BATCH_SIZE = _env_int("MEMORY_BATCH_SIZE", 50)             # 每批回放的行为事件数（水位批粒度推进）

# 去重与降权
EMBED_SIM_THRESHOLD = _env_float("MEMORY_EMBED_SIM_THRESHOLD", 0.85)  # 余弦相似度达该值视为同一条目（合并）
IMPORTANCE_DOWNWEIGHT_FACTOR = _env_float("MEMORY_DOWNWEIGHT_FACTOR", 0.5)  # 互斥类型冲突时旧条目重要度乘数
RETIRE_IMPORTANCE_THRESHOLD = _env_float("MEMORY_RETIRE_THRESHOLD", 0.25)  # 重要度低于该值置 retired

# 置信度与固化
CONFIDENCE_BASE = _env_float("MEMORY_CONFIDENCE_BASE", 0.3)       # 置信度基数
CONFIDENCE_PER_SESSION = _env_float("MEMORY_CONFIDENCE_PER_SESSION", 0.15)  # 每跨会话增量
CONFIDENCE_DECAY_PER_DAY = _env_float("MEMORY_CONFIDENCE_DAILY_DECAY", 0.9)  # 每日时间衰减系数
PROMOTE_THRESHOLD = _env_float("MEMORY_PROMOTE_THRESHOLD", 0.6)   # 置信度达该值固化写 UserPreference

# 互斥类型：同类型新证据与旧条目语义不相似时，对旧条目执行冲突降权
# service/duration 不互斥（用户可有多种偏好），降权只作用于互斥类型
MUTUALLY_EXCLUSIVE_TYPES = {"time_period", "technician"}

# ============ C4b 分层记忆：加权召回与会话窗口 ============

# 加权召回：得分 = 语义 0.6 + 时效 0.3 + 重要度 0.1，取 Top-K 注入上下文
RECALL_WEIGHT_SEMANTIC = _env_float("MEMORY_RECALL_W_SEMANTIC", 0.6)
RECALL_WEIGHT_RECENCY = _env_float("MEMORY_RECALL_W_RECENCY", 0.3)
RECALL_WEIGHT_IMPORTANCE = _env_float("MEMORY_RECALL_W_IMPORTANCE", 0.1)
RECALL_TOP_K = _env_int("MEMORY_RECALL_TOP_K", 5)

# 会话窗口：短期记忆最近 10 轮（每轮 user+assistant，最多 20 条消息）
WINDOW_MESSAGE_LIMIT = _env_int("MEMORY_WINDOW_MESSAGE_LIMIT", 20)
# 占用达窗口 60% 时触发滚动摘要，压缩最早消息
WINDOW_SUMMARY_TRIGGER_RATIO = _env_float("MEMORY_WINDOW_SUMMARY_RATIO", 0.6)
# 压缩后保留的最近消息数（窗口的 40%）
WINDOW_KEEP_AFTER_SUMMARY = _env_int(
    "MEMORY_WINDOW_KEEP_AFTER_SUMMARY",
    int(WINDOW_MESSAGE_LIMIT * (1 - WINDOW_SUMMARY_TRIGGER_RATIO)),
)
