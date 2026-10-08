"""
预约 Agent 离线测试（C1 重写）

原 7 个用例基线即失败：依赖真实 LLM（parse_stream 同步接口），
且引用已删除的旧属性（appointment_history 挂实例、agent.reset()）。
按当前架构重写：
- 会话状态挂 SessionContext（实例无状态）
- LLM 经 monkeypatch create_chat_model 注入假实现，零真实 API 调用
- 与 test_structured_output（解析层）与 test_booking_confirmation（落库前确认）互补，
  本文件聚焦 Agent 主流程 run_stream 的编排行为
"""
import asyncio

import pytest

from agents.appointment.appointment_processor import AppointmentProcessor
from agents.appointment.message_builder import MessageBuilder
from agents.appointment.schemas import AppointmentExtraction
from agents.appointment_agent import AppointmentAgent
from agents.core.session import SessionContext, new_appointment_history
from config.constants import StateEnum
from tests.conftest import FakeStructuredLLM


def _make_agent(monkeypatch, llm):
    """构造 AppointmentAgent：LLM 经 create_chat_model 注入；
    天气已改为工具直调（无 ReAct agent），测试路径不触发天气调用"""
    monkeypatch.setattr("agents.appointment_agent.create_chat_model", lambda **kw: llm)
    return AppointmentAgent()


def _extraction(**overrides) -> AppointmentExtraction:
    """构造抽取结果（默认与预约无关、信息不完整）"""
    base = dict(
        gender="未知", start_time="未知", duration="未知", project="未知",
        preference="未知", technician_name="未知", confirmation="未知",
        info_complete=False, unrelated=False, missing_info=[],
    )
    base.update(overrides)
    return AppointmentExtraction(**base)


async def _collect(agen):
    return [t async for t in agen]


def test_extract_and_fill_slots_from_natural_language(monkeypatch):
    """自然语言输入 → 结构化抽取 → 槽位落入会话预约历史

    抽取缺时间（必填槽位），流程停留在引导路径：完成路径涉及真实
    理疗师库查询，其分支行为由 test_booking_confirmation 覆盖。
    """
    agent = _make_agent(monkeypatch, FakeStructuredLLM(_extraction(
        gender="女", project="肩颈理疗",
    )))
    ctx = SessionContext(session_id="s1")

    tokens = asyncio.run(_collect(agent.run_stream("我想预约明天下午2点的理疗，女理疗师", session=ctx)))

    assert ctx.appointment_history["project"] == "肩颈理疗"
    assert ctx.appointment_history["gender"] == "女"
    assert any("缺少" in t for t in tokens), "缺失时间应触发引导语"
    assert len(tokens) > 0, "应产出响应内容"


def test_partial_slots_merge_keeps_previous_values():
    """分步补槽：必填=时间+项目；时长缺省按项目标准时长补齐"""
    processor = AppointmentProcessor(None, None, MessageBuilder(), None, None)
    history = new_appointment_history()

    # 第一轮只提供项目（缺时间，必填未齐）
    finished1 = processor.update_history_from_data(history, {"project": "肩颈理疗"})
    assert finished1 is False, "缺时间不应视为完成"
    assert history["project"] == "肩颈理疗"
    assert history["start_time"] is None

    # 第二轮补充时间 → 完成，且时长按项目标准时长自动补齐
    finished2 = processor.update_history_from_data(
        history, {"start_time": "2026-09-24 14:00"})
    assert history["project"] == "肩颈理疗", "已填槽位应保持"
    assert history["start_time"] == "2026-09-24 14:00"
    assert finished2 is True
    assert history["duration"] == "60分钟", "肩颈理疗标准时长 60 分钟应自动补齐"

    # 性别为选填槽位，不再影响完成判定
    assert history["gender"] is None


def test_unrelated_input_routes_to_callback_and_keeps_slots(monkeypatch):
    """预约中途插话（已有槽位）：输出标准拒绝话术，状态保持 APPOINTMENT，槽位保留

    对应 e2e-010：插话无关后槽位保留并顺利完成预约——不回退分类状态，
    避免下一轮"肩颈理疗"被当成独立咨询任务。
    """
    calls = {}

    async def fake_callback(user_input, session):
        calls["input"] = user_input
        calls["session"] = session
        return "已转交"

    agent = _make_agent(monkeypatch, FakeStructuredLLM(_extraction(unrelated=True)))
    agent.unrelated_callback = fake_callback
    ctx = SessionContext(session_id="s1")
    ctx.appointment_history["project"] = "肩颈理疗"  # 此前已填的槽位
    ctx.state.value = StateEnum.APPOINTMENT

    tokens = asyncio.run(_collect(agent.run_stream("今天天气怎么样？", session=ctx)))

    # 有槽位时不转交回调、不重置状态：直接输出标准话术
    assert not calls, "有槽位时不应转交归类机器人"
    assert ctx.state.value == StateEnum.APPOINTMENT
    assert ctx.appointment_history["project"] == "肩颈理疗", "无关请求不应清空已填槽位"
    assert any("暂不支持" in t and "健康理疗" in t for t in tokens)


def test_unrelated_input_without_slots_routes_to_callback(monkeypatch):
    """无槽位（预约流程未开始）的无关请求：转交归类机器人，状态回 CLASSIFY"""
    calls = {}

    async def fake_callback(user_input, session):
        calls["input"] = user_input
        calls["session"] = session
        return "已转交"

    agent = _make_agent(monkeypatch, FakeStructuredLLM(_extraction(unrelated=True)))
    agent.unrelated_callback = fake_callback
    ctx = SessionContext(session_id="s1")
    ctx.state.value = StateEnum.APPOINTMENT

    tokens = asyncio.run(_collect(agent.run_stream("今天天气怎么样？", session=ctx)))

    assert calls["input"] == "今天天气怎么样？"
    assert ctx.state.value == StateEnum.CLASSIFY


def test_incomplete_info_asks_for_missing_fields(monkeypatch):
    """信息不完整：输出缺失字段引导语，不进入完成流程"""
    agent = _make_agent(monkeypatch, FakeStructuredLLM(_extraction(project="肩颈理疗")))
    ctx = SessionContext(session_id="s1")

    tokens = asyncio.run(_collect(agent.run_stream("我想预约肩颈理疗", session=ctx)))

    thought = next(t for t in tokens if "缺少" in t)
    assert "start_time" in thought
    assert "duration" not in thought, "时长已改为选填，不再追问"
    assert any("[REPLY][预约机器人]" in t for t in tokens)


def test_parse_failure_degrades_to_default_not_crash(monkeypatch):
    """LLM 抽取失败：降级为默认抽取结果，输出引导语而非崩溃"""
    class RaisingLLM:
        def with_structured_output(self, schema, method=None):
            return None

        async def ainvoke(self, messages):
            raise RuntimeError("模拟 LLM 不可用")

    agent = _make_agent(monkeypatch, RaisingLLM())
    ctx = SessionContext(session_id="s1")

    tokens = asyncio.run(_collect(agent.run_stream("预约", session=ctx)))

    assert len(tokens) > 0, "降级后仍应有响应"
    assert any("[REPLY][预约机器人]" in t for t in tokens)


def test_reset_after_appointment_restores_initial_state(monkeypatch):
    """预约完成后重置：槽位清空、状态回 CLASSIFY、聊天历史清空"""
    agent = _make_agent(monkeypatch, FakeStructuredLLM(_extraction()))
    ctx = SessionContext(session_id="s1")
    ctx.appointment_history["project"] = "肩颈理疗"
    ctx.appointment_history["start_time"] = "2026-09-24 14:00"
    ctx.state.value = StateEnum.APPOINTMENT
    ctx.chat_history.add_user_message("已预约")

    agent._reset_after_appointment(ctx)

    assert ctx.appointment_history == new_appointment_history()
    assert ctx.state.value == StateEnum.CLASSIFY
    assert len(ctx.chat_history.messages) == 0


def test_appointment_history_lives_on_session_not_agent(monkeypatch):
    """回归护栏：预约状态挂会话上下文，Agent 实例不再携带 appointment_history"""
    agent = _make_agent(monkeypatch, FakeStructuredLLM(_extraction()))
    assert not hasattr(agent, "appointment_history"), (
        "旧架构的实例级 appointment_history 已移除，状态应挂 SessionContext")
    assert not hasattr(agent, "finished"), "旧架构的实例级 finished 已移除"
