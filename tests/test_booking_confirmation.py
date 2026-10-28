"""
落库前确认与禁忌筛查测试（B4 验收）

全部离线。覆盖：
1. 确认回复四分支（禁忌命中/确认/取消/模糊）的状态机逻辑
2. handle_complete_appointment 各分支的信号与消息
3. 图节点路由：confirmation_pending → confirm 节点输出摘要问询
4. manual_review 节点：禁忌转人工话术 + 会话重置
5. 会话状态路由：预约流程中"确认"回复不走 LLM 分类，直接进入预约节点
"""
import asyncio
import pytest

from agents.appointment.appointment_processor import AppointmentProcessor
from agents.appointment.message_builder import MessageBuilder
from agents.core.session import SessionContext
from agents.graph import build_agent_graph, stream_graph_task
from config.constants import StateEnum


def make_processor(emit_confirm_signal=False):
    """构造 AppointmentProcessor：llm=None 避免天气工具初始化"""
    processor = AppointmentProcessor(
        input_parser=None, technician_finder=None, message_builder=MessageBuilder(),
        appointment_database=None, llm=None
    )
    processor.emit_confirm_signal = emit_confirm_signal
    return processor


# ============ 状态机单元测试 ============

def test_contraindication_reply_sets_hit_and_completes():
    """禁忌命中：记录禁忌词并标记流程完成"""
    processor = make_processor()
    history = {"awaiting_booking_confirmation": True, "pending_technician": {"name": "张伟"}}

    finished = processor.update_history_from_data(
        history, {"confirmation": "未知"}, user_input="我有高血压"
    )

    assert finished is True
    assert history["contraindication_hit"] == ["高血压"]
    assert "awaiting_booking_confirmation" not in history or not history["awaiting_booking_confirmation"]


def test_positive_reply_confirms_booking():
    """肯定回复：booking_confirmed 置 True，流程完成"""
    processor = make_processor()
    history = {"awaiting_booking_confirmation": True, "pending_technician": {"name": "张伟"}}

    finished = processor.update_history_from_data(
        history, {"confirmation": "可以"}, user_input="好的，确认"
    )

    assert finished is True
    assert history["booking_confirmed"] is True


def test_negative_reply_cancels_booking():
    """否定回复：booking_cancelled 置 True，流程完成"""
    processor = make_processor()
    history = {"awaiting_booking_confirmation": True, "pending_technician": {"name": "张伟"}}

    finished = processor.update_history_from_data(
        history, {"confirmation": "不要"}, user_input="算了，取消吧"
    )

    assert finished is True
    assert history["booking_cancelled"] is True


def test_unclear_reply_keeps_waiting():
    """模糊回复：保持等待确认，流程不完成"""
    processor = make_processor()
    history = {"awaiting_booking_confirmation": True, "pending_technician": {"name": "张伟"}}

    finished = processor.update_history_from_data(
        history, {"confirmation": "未知"}, user_input="嗯..."
    )

    assert finished is False
    assert history.get("awaiting_booking_confirmation") is True


# ============ handle_complete_appointment 分支 ============

def _collect(agen):
    return asyncio.run(_collect_async(agen))


async def _collect_async(agen):
    return [t async for t in agen]


def test_complete_appointment_emits_confirm_signal_in_graph_mode():
    """图模式 + 禁忌命中：完成流程只发出 manual_review 信号"""
    processor = make_processor(emit_confirm_signal=True)
    history = {"contraindication_hit": ["孕期"]}

    tokens = _collect(processor.handle_complete_appointment(history, "s1"))

    assert tokens == ["[SIGNAL]manual_review"]


def test_complete_appointment_cancelled_yields_cancel_message():
    """已取消：输出取消话术并清理 pending_technician"""
    processor = make_processor()
    history = {"booking_cancelled": True, "pending_technician": {"name": "张伟"}}

    tokens = _collect(processor.handle_complete_appointment(history, "s1"))

    assert any("已为您取消本次预约" in t for t in tokens)
    assert "pending_technician" not in history


def test_complete_appointment_confirmed_saves_with_pending_tech():
    """已确认：经假落库保存 pending 技师，输出预约成功话术"""
    processor = make_processor()
    saved = {}

    async def fake_success(tech, history, session_id):
        saved["tech"] = tech
        return "预约成功"

    processor._process_successful_appointment = fake_success
    history = {"booking_confirmed": True, "pending_technician": {"id": 1, "name": "张伟"}}

    tokens = _collect(processor.handle_complete_appointment(history, "s1"))

    assert saved["tech"]["name"] == "张伟"
    assert any("预约成功" in t for t in tokens)


# ============ 图节点路由测试 ============

class StubConsultant:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def consult_stream(self, user_input, session=None):
        yield "[REPLY][咨询机器人]stub"


class TwoTurnAppointmentAgent:
    """两轮预约流：第一轮发出确认信号，第二轮确认后成功"""

    def __init__(self):
        self.llm = None
        self.message_builder = MessageBuilder()
        self.turns = 0

    async def run_stream(self, user_input=None, session=None,
                         enable_replan_signal=False, enable_confirm_signal=False):
        self.turns += 1
        if self.turns == 1:
            session.appointment_history["pending_technician"] = {"id": 1, "name": "张伟", "gender": "男"}
            session.appointment_history["awaiting_booking_confirmation"] = True
            yield "[SIGNAL]confirmation_pending"
        else:
            yield "[REPLY][预约机器人]预约成功！"


class ManualReviewAppointmentAgent:
    """禁忌命中流：发出 manual_review 信号并在会话中留存命中信息"""

    def __init__(self):
        self.llm = None
        self.message_builder = MessageBuilder()

    async def run_stream(self, user_input=None, session=None,
                         enable_replan_signal=False, enable_confirm_signal=False):
        session.appointment_history["contraindication_hit"] = ["高血压"]
        session.appointment_history["pending_technician"] = {"id": 1, "name": "张伟"}
        yield "[SIGNAL]manual_review"


class StubClassifier:
    """脚本化分类器：首轮返回 appointment（进入预约流），后续返回 other（验证状态机路由优先）"""

    def __init__(self, responses=None):
        self.responses = list(responses) if responses else ["appointment", "other"]
        self.calls = 0

    async def classify_task(self, task):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


class StubTaskAgent:
    def __init__(self, appointment_agent, db_url):
        self.task_classifier = StubClassifier()
        self.appointment_agent = appointment_agent
        self.consultant_agent = StubConsultant()
        self.behavior_agent = None
        from agents.core.session import SessionRegistry
        self.session_registry = SessionRegistry(db_path=db_url)


async def _run_turn(tmp_path, task_agent, sid, user_input):
    graph = build_agent_graph(task_agent)
    tokens = []
    async for token in stream_graph_task(graph, user_input, sid):
        tokens.append(token)
    return "".join(tokens)


def test_confirmation_pending_routes_to_confirm_node(tmp_path):
    """确认信号 → confirm 节点输出预约摘要与禁忌问询"""
    agent = TwoTurnAppointmentAgent()
    task_agent = StubTaskAgent(agent, f"sqlite:///{tmp_path}/g.db")

    joined = asyncio.run(_run_turn(tmp_path, task_agent, "s1", "预约理疗"))

    assert "请您确认预约信息" in joined
    assert "张伟" in joined
    assert "高血压、孕期、皮肤破损" in joined


def test_state_routing_keeps_booking_flow_without_llm_classify(tmp_path):
    """预约流程中：分类器返回 other，但状态机在 APPOINTMENT → 仍进入预约节点完成落库"""
    agent = TwoTurnAppointmentAgent()
    task_agent = StubTaskAgent(agent, f"sqlite:///{tmp_path}/g.db")

    joined1 = asyncio.run(_run_turn(tmp_path, task_agent, "s1", "预约理疗"))
    assert "请您确认预约信息" in joined1

    # 第二轮：用户回复"确认"；分类器 stub 返回 other，靠状态机路由
    joined2 = asyncio.run(_run_turn(tmp_path, task_agent, "s1", "确认"))
    assert "预约成功" in joined2


def test_manual_review_routes_to_node_and_resets_session(tmp_path):
    """禁忌信号 → manual_review 节点：转人工话术 + 会话槽位重置 + 状态回 CLASSIFY"""
    agent = ManualReviewAppointmentAgent()
    task_agent = StubTaskAgent(agent, f"sqlite:///{tmp_path}/g.db")
    sid = "s1"

    joined = asyncio.run(_run_turn(tmp_path, task_agent, sid, "我有高血压"))

    assert "可能不适合直接安排理疗服务" in joined
    assert "高血压" in joined

    session = task_agent.session_registry.get_or_create(sid)
    assert session.state.value == StateEnum.CLASSIFY
    assert session.appointment_history.get("contraindication_hit") is None
    assert session.appointment_history.get("pending_technician") is None
    assert session.appointment_history.get("project") is None


def test_confirmation_message_and_referral_message_content():
    """话术文案：确认摘要含技师/项目/时间，禁忌转介含禁忌词"""
    builder = MessageBuilder()
    history = {"project": "肩颈理疗", "start_time": "2026-09-23 15:00", "duration": "60分钟"}
    msg = builder.create_booking_confirmation_message({"name": "张伟"}, history)
    assert "张伟" in msg and "肩颈理疗" in msg and "2026-09-23 15:00" in msg

    referral = builder.create_contraindication_referral_message(["孕期", "高血压"])
    assert "孕期、高血压" in referral and "咨询医师" in referral
