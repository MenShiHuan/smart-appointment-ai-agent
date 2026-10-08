"""
推荐理疗师确认/拒绝链路离线测试（e2e-002/003 根因回归护栏）

覆盖 _handle_recommendation_response 四类输入：
- 明确同意 → 确认推荐技师
- 纯拒绝 → recommendation_declined
- 拒绝 + 换技师 → 合并新技师并清理推荐状态，继续正常匹配
- 拒绝 + 改时间 → 合并新时间（保留原技师），继续正常匹配
- 回应不明确 → False（由 run_stream 输出重新确认话术，而非"缺少："空话）
"""
import asyncio

from agents.appointment.appointment_processor import AppointmentProcessor
from agents.appointment.schemas import AppointmentExtraction


def _processor():
    return AppointmentProcessor(None, None, None, None, llm=None)


def _data(**overrides) -> dict:
    base = dict(
        gender="未知", start_time="未知", duration="未知", project="未知",
        preference="未知", technician_name="未知", confirmation="未知",
        info_complete=False, unrelated=False, missing_info=[],
    )
    base.update(overrides)
    return AppointmentExtraction(**base).model_dump()


def _history_with_recommendation():
    return {
        "gender": None, "start_time": "2026-09-25 15:00", "duration": "60分钟",
        "project": "艾灸", "preference": None, "technician_name": "张伟",
        "awaiting_confirmation": True,
        "recommended_technician": {"id": 2, "name": "王强", "gender": "男"},
        "original_technician": {"id": 1, "name": "张伟", "gender": "男"},
    }


def test_positive_response_confirms_recommendation():
    proc = _processor()
    history = _history_with_recommendation()
    result = proc._handle_recommendation_response(history, _data(confirmation="好"))
    assert result is True
    assert history["confirmed_technician"]["name"] == "王强"
    assert history["awaiting_confirmation"] is False


def test_negative_response_without_new_choice_declines():
    proc = _processor()
    history = _history_with_recommendation()
    result = proc._handle_recommendation_response(history, _data(confirmation="不"))
    assert result is True
    assert history["recommendation_declined"] is True
    assert "confirmed_technician" not in history


def test_negative_response_with_new_technician_merges_slots():
    """e2e-002：'那换成李娜老师吧' → 拒绝 + 换技师 → 合并并清理推荐状态"""
    proc = _processor()
    history = _history_with_recommendation()
    result = proc._handle_recommendation_response(
        history, _data(confirmation="换成李娜老师吧", technician_name="李娜"))
    assert result is True
    assert history["technician_name"] == "李娜"
    assert "recommended_technician" not in history
    assert "original_technician" not in history
    assert "recommendation_declined" not in history
    assert history["awaiting_confirmation"] is False


def test_negative_response_with_new_time_keeps_technician():
    """e2e-003：'我不换别人，那我改到后天下午三点吧' → 保留原技师，更新时间为准"""
    proc = _processor()
    history = _history_with_recommendation()
    result = proc._handle_recommendation_response(
        history, _data(confirmation="不换别人，改到后天下午三点", start_time="2026-09-26 15:00"))
    assert result is True
    assert history["technician_name"] == "张伟"  # 原技师保留
    assert history["start_time"] == "2026-09-26 15:00"
    assert "recommendation_declined" not in history


def test_unclear_response_returns_false():
    proc = _processor()
    history = _history_with_recommendation()
    result = proc._handle_recommendation_response(history, _data(confirmation="未知"))
    assert result is False
    assert history["awaiting_confirmation"] is True  # 继续等待明确回复


def test_incomplete_info_guard_emits_reconfirm_not_empty_missing():
    """槽位齐全时兜底走重新确认话术，不再输出'缺少：，'病句"""
    async def collect():
        proc = _processor()
        from agents.appointment.message_builder import MessageBuilder
        proc.message_builder = MessageBuilder()
        history = {
            "gender": None, "start_time": "2026-09-25 15:00", "duration": "60分钟",
            "project": "艾灸", "preference": None, "technician_name": None,
        }
        tokens = []
        async for t in proc.handle_incomplete_info({}, history):
            tokens.append(t)
        return "".join(tokens)

    joined = asyncio.run(collect())
    assert "缺少：" not in joined
    assert "确认" in joined


def test_atomic_save_rejects_conflicting_schedule():
    """事务内冲突复查：已占用时段返回 None，不落库"""
    from db.db_router import DatabaseRouter
    from db.models import TechnicianSchedule
    from datetime import datetime

    db = DatabaseRouter()
    start = datetime(2026, 10, 1, 10, 0)
    end = datetime(2026, 10, 1, 11, 0)
    repo = db.technician_repo

    # 清理残留
    with db.session_manager.session_scope() as session:
        session.query(TechnicianSchedule).filter(
            TechnicianSchedule.start_time == start).delete(synchronize_session=False)

    first = repo.save_appointment_if_available(1, start, end, "busy", appointment_id=100)
    assert first is not None
    conflict = repo.save_appointment_if_available(1, start, end, "busy", appointment_id=101)
    assert conflict is None  # 双订被拒绝

    # 不重叠时段可落库
    ok = repo.save_appointment_if_available(
        1, end, datetime(2026, 10, 1, 12, 0), "busy", appointment_id=102)
    assert ok is not None

    # 清理
    with db.session_manager.session_scope() as session:
        session.query(TechnicianSchedule).filter(
            TechnicianSchedule.appointment_id.in_([100, 101, 102])).delete(
            synchronize_session=False)
