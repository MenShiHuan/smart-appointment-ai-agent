# services/text_embedding.py

import math

import numpy as np
import faiss
import threading
from config.model_provider import create_embedding_model


# ============ embedding 客户端单例（D1） ============
# embed_input 高频调用（检索/记忆整合/重嵌入），每次新建客户端浪费初始化开销；
# 模块级懒加载单例（双检锁），全进程共享一个 Ollama bge-m3 客户端。

_embedding_model = None
_embedding_lock = threading.Lock()


def get_embedding_model():
    """获取进程级 embedding 客户端单例（懒加载，线程安全）"""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                _embedding_model = create_embedding_model()
    return _embedding_model


def _normalize(vector) -> list:
    """向量归一化：内积在归一化后等价于余弦相似度（与知识库/记忆检索口径一致）"""
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


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
    """文本 → bge-m3 向量（进程级客户端单例）"""
    embeddings = get_embedding_model()
    return embeddings.embed_query(input_text)
