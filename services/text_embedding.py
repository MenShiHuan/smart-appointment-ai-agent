"""
文本嵌入工具（全链路向量化统一 DashScope text-embedding-v3，1024 维）

原 Ollama bge-m3 本地方案已下线（Ollama 已卸载）。本模块供
记忆模块（SQLite 向量存储/加权召回）与技师专长匹配使用；
知识 RAG 的编码走 Milvus 服务端 TEXTEMBEDDING Function（同一模型）。
凭证来自 .env 的 DASHSCOPE_API_KEY。
"""
# services/text_embedding.py

import math
import threading

import numpy as np
import faiss
import requests

from config import vector_config as vc


# ============ HTTP 会话单例 ============
# embed_input 高频调用（记忆整合/召回），复用连接避免每次握手开销。

_session = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
    return _session


def normalize_vector(vector) -> list:
    """向量归一化：内积在归一化后等价于余弦相似度（与知识库/记忆检索口径一致）"""
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


# 私有别名：模块内既有调用保持
_normalize = normalize_vector


def find_best_match_indices(text: str, candidates: list) -> list:
    """
    通过 embedding 相似度返回所有候选项的索引，按相似度从高到低排序

    :param text: 待匹配文本
    :param candidates: 候选文本列表
    :return: 所有候选项的索引列表，按相似度从高到低排序
    """
    if not candidates:
        return []
    candidate_embs = np.array([_normalize(embed_input(c)) for c in candidates]).astype("float32")
    dimension = candidate_embs.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 归一化向量内积 = 余弦相似度
    index.add(candidate_embs)
    text_emb = np.array([_normalize(embed_input(text))]).astype("float32")
    k = len(candidates)
    _, indices = index.search(text_emb, k)
    return indices[0][:k].tolist()


def embed_input(input_text: str) -> list:
    """文本 → text-embedding-v3 稠密向量（1024 维，DashScope 云 API）"""
    if not vc.DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置：向量化需要 .env 中的 DashScope 凭证")
    resp = _get_session().post(
        vc.DASHSCOPE_EMBED_URL,
        json={
            "model": vc.EMBED_MODEL,
            "input": input_text,
            "dimensions": vc.DENSE_DIM,
        },
        headers={"Authorization": f"Bearer {vc.DASHSCOPE_API_KEY}"},
        timeout=60,
    )
    resp.raise_for_status()
    return [float(x) for x in resp.json()["data"][0]["embedding"]]
