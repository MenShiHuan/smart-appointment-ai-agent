"""
知识库检索参数配置（D1）

向量已归一化（IndexFlatIP 内积即余弦相似度），分数范围 [-1, 1]。
"""
import os


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


# 检索候选放大倍数：先取 top_k × 该倍数再过滤/截断（分类过滤后仍能凑满 top_k）
CANDIDATE_MULTIPLIER = _env_int("KNOWLEDGE_CANDIDATE_MULTIPLIER", 2)

# 相似度下限：低于该余弦相似度的候选不注入提示词（防无关知识干扰 LLM）
MIN_SIMILARITY = _env_float("KNOWLEDGE_MIN_SIMILARITY", 0.3)
