"""
离线记忆整合测试（C4a 验收）

全部离线：FakeLLM/FakeEmbedding + 临时 SQLite，零真实 API 调用。
覆盖 DEV_SPEC 5.4 C4a 验收清单：
1. 触发条件判定（会话数 / 间隔 / 新增会话数）
2. 增量回放只处理水位之后的事件；重复运行幂等
3. 规则抽取（appointment 行为）
4. LLM 抽取（consultation 行为）+ 失败不阻塞
5. 向量去重合并（evidence_count / distinct_sessions 递增）
6. 互斥类型冲突降权与退役
7. 置信度公式与固化阈值（≥0.6 写 UserPreference）
8. 批粒度水位推进与断点续跑
"""
import asyncio
import hashlib
import math
from datetime import datetime, timedelta

import pytest

from agents.user_behavior.memory_extractor import (
    PreferenceCandidate,
    PreferenceExtractionResult,
    ExtractedPreference,
    extract_from_behavior,
)
from db.db_router import DatabaseRouter
from db.models import utcnow
from services.memory_service import MemoryService
from tests.conftest import FakeLLM, FakeStructuredLLM


# ============ 确定性假 embedding ============

class DictEmbedding:
    """注册文本 → 正交单位向量（相似度精确可控）；未注册文本 → 哈希向量"""

    def __init__(self, known_texts):
        self.known = {}
        for i, text in enumerate(known_texts):
            vec = [0.0] * len(known_texts)
            vec[i] = 1.0
            self.known[text] = vec
        self.cache = dict(self.known)

    def __call__(self, text):
        if text in self.cache:
            return self.cache[text]
        digest = hashlib.md5(text.encode("utf-8")).digest()
        dim = max(4, len(self.known))
        vec = [0.0] * dim
        for i, byte in enumerate(digest):
            vec[i % dim] += (byte / 255.0 - 0.5) * 2
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        self.cache[text] = vec
        return vec


EMBEDDING_TEXTS = ["肩颈理疗", "下午", "上午", "晚上", "60分钟", "1", "2", "3", "4", "5", "足部理疗"]


@pytest.fixture
def db_path(tmp_path):
    return f"sqlite:///{tmp_path}/c4a_memory.db"


@pytest.fixture
def memory_service(db_path, monkeypatch):
    """构造 MemoryService：临时库 + 假 LLM + 确定性 embedding"""
    fake_llm = FakeLLM(["noop"])
    monkeypatch.setattr("services.memory_service.embed_input", DictEmbedding(EMBEDDING_TEXTS))
    return MemoryService(db_path=db_path, llm=fake_llm)


def _record(memory_service, session_id, project="肩颈理疗", start_time="2026-09-20 14:00",
            duration="60分钟", technician_id=None, action_type="appointment",
            question=None):
    """写入一条行为事件，返回 behavior id"""
    return memory_service.user_repo.record_behavior(
        user_id="default_user",
        action_type=action_type,
        action_data=({"project": project, "start_time": start_time,
                      "duration": duration} if action_type == "appointment"
                     else ({"question": question} if question else {})),
        technician_id=technician_id,
        session_id=session_id,
    )


def _seed_sessions(memory_service, sessions, technician_ids=None):
    """按会话列表批量写入 appointment 行为"""
    for i, sid in enumerate(sessions):
        _record(memory_service, sid, technician_id=(technician_ids[i] if technician_ids else None))


# ============ 1. 触发条件判定 ============

def test_trigger_requires_min_sessions(memory_service):
    now = utcnow()
    should, reason = memory_service.check_trigger(None, 3, now)
    assert should is False
    assert "会话数不足" in reason

    should, _ = memory_service.check_trigger(None, 5, now)
    assert should is True


def test_trigger_requires_new_sessions_and_interval(memory_service):
    now = utcnow()
    wm = {"last_behavior_id": 10,
          "last_integration_at": now - timedelta(hours=12),
          "session_count_at_last_run": 10}

    # 新增会话不足
    should, reason = memory_service.check_trigger(wm, 13, now)
    assert should is False
    assert "新增会话不足" in reason

    # 间隔不足 24h
    should, reason = memory_service.check_trigger(wm, 15, now)
    assert should is False
    assert "间隔不足" in reason

    # 条件满足
    should, _ = memory_service.check_trigger(wm, 15, now + timedelta(hours=25))
    assert should is True


def test_run_not_triggered_when_below_threshold(memory_service):
    _seed_sessions(memory_service, ["s1", "s2"])
    report = asyncio.run(memory_service.run_integration_if_due())

    assert report["triggered"] is False
    assert "会话数不足" in report["reason"]
    assert memory_service.memory_repo.get_watermark("default_user") is None


# ============ 3. 规则抽取 ============

def test_rule_extraction_from_appointment():
    behavior = {"id": 1, "action_type": "appointment", "session_id": "s1",
                "technician_id": 7,
                "action_data": {"project": "肩颈理疗", "start_time": "2026-09-20 14:00",
                                "duration": "60分钟", "preference": "手法轻"}}

    candidates = extract_from_behavior(behavior)

    by_type = {c.entry_type: c for c in candidates}
    assert by_type["service"].content == "肩颈理疗"
    assert by_type["time_period"].content == "下午"
    assert by_type["duration"].content == "60分钟"
    assert by_type["technician"].content == "7"
    assert by_type["technician_type"].content == "手法轻"
    assert all(c.behavior_id == 1 and c.session_id == "s1" for c in candidates)


# ============ 4. LLM 抽取 + 失败不阻塞 ============

def test_llm_extraction_from_consultation(db_path, monkeypatch):
    result = PreferenceExtractionResult(preferences=[
        ExtractedPreference(preference_type="service", content="足部理疗")])
    structured_llm = FakeStructuredLLM(result)
    monkeypatch.setattr("services.memory_service.embed_input",
                        DictEmbedding(EMBEDDING_TEXTS))
    service = MemoryService(db_path=db_path, llm=structured_llm)
    _seed_sessions(service, ["s1", "s2", "s3", "s4"])
    _record(service, "s5", action_type="consultation", question="我想试试足部理疗")

    report = asyncio.run(service.run_integration_if_due())

    assert report["triggered"] is True
    entries = service.memory_repo.get_active_entries("default_user", "service")
    assert any(e["content"] == "足部理疗" for e in entries)


def test_llm_extraction_failure_does_not_block(db_path, monkeypatch):
    class RaisingLLM:
        def with_structured_output(self, schema, method=None):
            return None

        async def ainvoke(self, messages):
            raise RuntimeError("模拟 LLM 不可用")

    monkeypatch.setattr("services.memory_service.embed_input",
                        DictEmbedding(EMBEDDING_TEXTS))
    failing_service = MemoryService(db_path=db_path, llm=RaisingLLM())
    _seed_sessions(failing_service, ["s1", "s2", "s3", "s4"])
    _record(failing_service, "s5", action_type="consultation", question="我腰痛怎么办？")

    report = asyncio.run(failing_service.run_integration_if_due())

    assert report["triggered"] is True
    # 规则抽取的 appointment 候选照常处理，LLM 失败只丢弃 consultation 候选
    entries = failing_service.memory_repo.get_active_entries("default_user", "service")
    assert any(e["content"] == "肩颈理疗" for e in entries)


# ============ 5/7. 去重合并 + 置信度固化 ============

def test_dedupe_merge_and_promote(memory_service):
    _seed_sessions(memory_service, ["s1", "s2", "s3", "s4", "s5"])  # 全为肩颈理疗/下午/60分钟

    report = asyncio.run(memory_service.run_integration_if_due())

    assert report["triggered"] is True
    service_entries = memory_service.memory_repo.get_active_entries("default_user", "service")
    assert len(service_entries) == 1, "相同服务文本应合并为单条"
    entry = service_entries[0]
    assert entry["evidence_count"] == 5
    assert entry["distinct_sessions"] == 5
    # 置信度 = min(1, 0.3 + 0.15×5) = min(1, 1.05) → 封顶 1.0，≥0.6 → 固化
    assert abs(entry["confidence"] - 1.0) < 1e-6
    # 每行为 3 个候选（service/time_period/duration）：首轮建 3 条，其余 12 次合并
    assert report["created"] == 3
    assert report["merged"] == 12

    prefs = memory_service.user_repo.get_user_preferences("default_user")
    by_type = {(p["preference_type"], p["preference_value"]) for p in prefs}
    assert ("service", "肩颈理疗") in by_type
    assert ("time_period", "下午") in by_type


# ============ 2/8. 幂等重跑 + 水位推进与断点续跑 ============

def test_idempotent_rerun_and_watermark_resume(memory_service, monkeypatch):
    monkeypatch.setattr("config.memory_config.BATCH_SIZE", 2)  # 批粒度 2，验证逐批水位推进
    now = utcnow()
    _seed_sessions(memory_service, ["s1", "s2", "s3", "s4", "s5"])

    report1 = asyncio.run(memory_service.run_integration_if_due(now=now))
    assert report1["triggered"] is True
    assert report1["processed_behaviors"] == 5

    wm = memory_service.memory_repo.get_watermark("default_user")
    last_id = max(b["id"] for b in memory_service.user_repo.get_user_behaviors("default_user"))
    assert wm["last_behavior_id"] == last_id, "水位应推进到最后一条已处理事件"

    # 幂等：无新事件时立即重跑 → 新增会话不足，不触发
    report2 = asyncio.run(memory_service.run_integration_if_due(now=now))
    assert report2["triggered"] is False
    entry = memory_service.memory_repo.get_active_entries("default_user", "service")[0]
    assert entry["evidence_count"] == 5, "重复运行不得重复计证据"

    # 断点续跑：新增 5 个会话后（间隔满足）只处理增量
    _seed_sessions(memory_service, ["s6", "s7", "s8", "s9", "s10"])
    report3 = asyncio.run(memory_service.run_integration_if_due(now=now + timedelta(hours=25)))
    assert report3["triggered"] is True
    assert report3["processed_behaviors"] == 5, "只应回放水位之后的新事件"
    entry = memory_service.memory_repo.get_active_entries("default_user", "service")[0]
    assert entry["evidence_count"] == 10
    assert entry["distinct_sessions"] == 10
    # 置信度封顶 1.0 且固化仍在
    assert abs(entry["confidence"] - 1.0) < 1e-6


# ============ 6. 互斥类型冲突降权与退役 ============

def test_conflict_downweight_and_retire(memory_service):
    # 5 个会话，理疗师各不相同：每出现新理疗师即对既有条目降权
    # tech1：1 → 0.5 → 0.25 → 0.125 退役；tech2：1 → 0.5 → 0.25 → 0.125 退役
    _seed_sessions(memory_service, ["s1", "s2", "s3", "s4", "s5"],
                   technician_ids=[1, 2, 3, 4, 5])

    report = asyncio.run(memory_service.run_integration_if_due())

    assert report["triggered"] is True
    all_entries = {e["content"]: e for e in _get_all_entries(
        memory_service, "default_user", "technician")}
    assert all_entries["1"]["status"] == "retired"
    assert abs(all_entries["1"]["importance"] - 0.125) < 1e-6
    assert all_entries["2"]["status"] == "retired"
    assert all_entries["3"]["status"] == "active"
    assert abs(all_entries["3"]["importance"] - 0.25) < 1e-6, "tech3 被降权两次仍活跃"
    assert all_entries["5"]["importance"] == 1.0, "最后出现的理疗师不被降权"

    # 变更日志记录降权与退役
    logs = [l for l in _get_change_logs(memory_service, "default_user")
            if l["operation"] in ("downweight", "retire")]
    assert len(logs) >= 4

    # 非互斥类型（service）不因新值降权：新增不同服务证据
    _seed_sessions(memory_service, ["s11", "s12", "s13", "s14"], technician_ids=[1, 1, 1, 1])
    _record(memory_service, "s15", project="足部理疗", technician_id=1)
    asyncio.run(memory_service.run_integration_if_due(now=utcnow() + timedelta(hours=49)))

    entries = {e["content"]: e for e in memory_service.memory_repo.get_active_entries("default_user", "service")}
    assert "足部理疗" in entries, "不同服务应建立新条目"
    assert entries["肩颈理疗"]["importance"] == 1.0, "非互斥类型不降权"


def _get_change_logs(service, user_id):
    from db.models import utcnow, MemoryChangeLog
    with service.memory_repo.session_manager.session_scope() as session:
        rows = session.query(MemoryChangeLog).filter(
            MemoryChangeLog.user_id == user_id).all()
        return [{"operation": r.operation, "entry_id": r.entry_id,
                 "detail": r.detail} for r in rows]


def _get_all_entries(service, user_id, entry_type=None):
    """查询全量条目（含 retired），供退役断言使用"""
    from db.models import utcnow, MemoryEntry
    with service.memory_repo.session_manager.session_scope() as session:
        query = session.query(MemoryEntry).filter(MemoryEntry.user_id == user_id)
        if entry_type:
            query = query.filter(MemoryEntry.entry_type == entry_type)
        return [{
            "content": e.content,
            "status": e.status,
            "importance": e.importance,
        } for e in query.all()]
