"""
知识库服务 D1 验收测试（RAG 稳定性与一致性小修）

全部离线。覆盖 DEV_SPEC 5.5 D1 验收清单：
1. 进程级单例：多次获取返回同一实例
2. API 新增知识后在线检索立即可见（单例索引自动刷新）
3. 检索批量回查（一次 IN 查询，无 N+1）
4. 向量归一化：检索分数为余弦相似度（≤1）
5. 相似度下限：低于阈值的候选不注入结果
6. 批量查询过滤软删除文档
7. initialize 并发安全（多调用方同时初始化不报错）
"""
import asyncio

import pytest

from db.db_router import DatabaseRouter
from services.knowledge_service import KnowledgeService, get_knowledge_service


class KeywordEmbedding:
    """关键词感知的确定性向量：命中正向词 → [1,0,...]；命中负向词 → [0,1,...]；否则零向量"""

    def __init__(self, positive_words, negative_words=(), dim=8):
        self.positive = positive_words
        self.negative = negative_words
        self.dim = dim

    def __call__(self, text):
        if any(w in text for w in self.positive) and not any(w in text for w in self.negative):
            return [1.0] + [0.0] * (self.dim - 1)
        if any(w in text for w in self.negative):
            return [0.0, 1.0] + [0.0] * (self.dim - 2)
        return [0.0] * self.dim


@pytest.fixture
def fake_embedding(monkeypatch):
    monkeypatch.setattr("services.knowledge_service.embed_input",
                        KeywordEmbedding(("肩颈",), ("足部",)))
    return monkeypatch


def test_singleton_returns_same_instance(tmp_path, monkeypatch):
    """进程级单例：多次获取返回同一实例（结束后还原全局实例状态）"""
    monkeypatch.setattr("services.knowledge_service._instance", None)
    try:
        svc1 = get_knowledge_service(db_path=f"sqlite:///{tmp_path}/k.db")
        svc2 = get_knowledge_service(db_path=f"sqlite:///{tmp_path}/k.db")

        assert svc1 is svc2, "进程级单例应返回同一实例"
    finally:
        monkeypatch.setattr("services.knowledge_service._instance", None)


def test_add_document_visible_to_search_immediately(tmp_path, fake_embedding):
    """API 新增知识 → 单例索引自动重建 → 在线检索立即可见（D1 核心修复）"""
    service = KnowledgeService(db_path=f"sqlite:///{tmp_path}/k.db")
    asyncio.run(service.initialize())

    added = asyncio.run(service.add_document("肩颈拉伸操教学视频", category="服务介绍"))

    assert added is True
    results = asyncio.run(service.search("肩颈拉伸", top_k=3))
    assert any("拉伸操" in doc["content"] for doc in results), "新增文档应立即进入检索结果"


def test_search_batches_document_fetch(tmp_path, fake_embedding):
    """检索回查走批量 IN 查询，非逐条 N+1"""
    service = KnowledgeService(db_path=f"sqlite:///{tmp_path}/k.db")
    asyncio.run(service.initialize())

    calls = {}
    repo = service.db
    original = repo.get_documents_by_ids

    def spy(ids):
        calls["count"] = calls.get("count", 0) + 1
        calls["ids"] = ids
        return original(ids)

    repo.get_documents_by_ids = spy
    results = asyncio.run(service.search("肩颈理疗", top_k=3))

    assert len(results) > 0
    assert calls["count"] == 1, "一次检索只应批量回查一次"
    assert len(calls["ids"]) > 1, "回查应携带全部候选 id"


def test_search_scores_are_cosine_bounded(tmp_path, fake_embedding):
    """归一化后分数为余弦相似度，不应超过 1"""
    service = KnowledgeService(db_path=f"sqlite:///{tmp_path}/k.db")
    asyncio.run(service.initialize())

    results = asyncio.run(service.search("肩颈理疗有什么好处", top_k=5))

    assert len(results) > 0
    for doc in results:
        assert 0 <= doc["score"] <= 1.0, f"分数越界: {doc['score']}"


def test_search_filters_below_similarity_threshold(tmp_path, fake_embedding):
    """低于相似度下限的候选不注入结果"""
    service = KnowledgeService(db_path=f"sqlite:///{tmp_path}/k.db")
    asyncio.run(service.initialize())
    # 种子知识含"足部理疗"条目：查询"肩颈"时其相似度为 0，应被阈值过滤
    asyncio.run(service.add_document("足部理疗手法介绍", category="服务介绍"))

    results = asyncio.run(service.search("肩颈理疗有什么好处", top_k=10))

    assert len(results) > 0
    assert all(doc["score"] >= 0.3 for doc in results), "低于阈值的结果不应出现"
    assert all("足部" not in doc["content"] for doc in results), "无关知识不应混入"


def test_batch_fetch_filters_soft_deleted(tmp_path):
    """get_documents_by_ids 过滤软删除文档"""
    db = DatabaseRouter(f"sqlite:///{tmp_path}/k.db")
    doc_id = db.knowledge.add_document("临时知识", "测试", [])
    db.knowledge.delete_document(doc_id, soft_delete=True)

    fetched = db.knowledge.get_documents_by_ids([doc_id])

    assert fetched == {}, "软删除文档不应被批量回查返回"


def test_initialize_concurrent_safe(tmp_path, fake_embedding):
    """多个调用方同时 initialize 不报错、不重复崩溃"""
    service = KnowledgeService(db_path=f"sqlite:///{tmp_path}/k.db")

    async def run():
        await asyncio.gather(service.initialize(), service.initialize(), service.initialize())

    asyncio.run(run())

    assert service.initialized is True
    assert service.index is not None
