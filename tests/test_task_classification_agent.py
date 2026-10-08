"""
任务分类 Agent 离线测试（C1 重写）

原 8 个用例依赖真实 LLM（4 个走真实 API 侥幸通过、4 个引用旧属性失败）。
重写要点：
- TaskClassifier 经 monkeypatch 替换为脚本化 stub（绕过 LCEL 对真实 LLM 的依赖）
- 下游 Agent 用 stub 替换，验证路由/状态机行为而非生成质量
- 会话挂临时 SQLite（SessionRegistry db_path 注入），不触碰 data/ 真实库
"""
import asyncio

import pytest

from agents.core.session import SessionRegistry
from agents.task_classification_agent import TaskClassificationAgent
from config.constants import StateEnum


class StubAppointmentAgent:
    def __init__(self):
        self.unrelated_callback = None

    async def run_stream(self, user_input=None, session=None,
                         enable_replan_signal=False, enable_confirm_signal=False):
        yield "[REPLY][预约机器人]stub-appointment"


class StubConsultant:
    def set_unrelated_callback(self, callback):
        self.unrelated_callback = callback

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def consult_stream(self, user_input, session=None):
        yield "[REPLY][咨询机器人]stub-consult"


class StubBehaviorAgent:
    async def answer_preference_query_stream(self, user_input):
        yield "[REPLY][记忆机器人]stub-memory"


def _make_agent(monkeypatch, tmp_path, categories, with_behavior=False):
    """构造分类 Agent：分类器按脚本返回类别，会话挂临时库，LLM 不参与"""
    class StubClassifier:
        """脚本化分类器：按调用次序返回预置类别"""

        def __init__(self, llm):
            self.responses = list(categories)
            self.calls = 0

        async def classify_task(self, task):
            idx = min(self.calls, len(self.responses) - 1)
            self.calls += 1
            return self.responses[idx]

    monkeypatch.setattr("agents.task_classification_agent.TaskClassifier", StubClassifier)
    monkeypatch.setattr("agents.task_classification_agent.create_chat_model", lambda **kw: object())

    agent = TaskClassificationAgent(
        StubAppointmentAgent(), StubConsultant(),
        behavior_agent=StubBehaviorAgent() if with_behavior else None,
    )
    agent.session_registry = SessionRegistry(db_path=f"sqlite:///{tmp_path}/c1_cls.db")
    return agent


async def _classify(agent, text, sid="s1"):
    return await agent.classify_task(text, session_id=sid)


def test_consultation_request_routes_to_consultant(monkeypatch, tmp_path):
    """咨询请求：分类为 query → 路由咨询机器人，状态 CONSULT"""
    agent = _make_agent(monkeypatch, tmp_path, ["query"])

    response = asyncio.run(_classify(agent, "肩颈理疗有什么好处？"))

    assert "stub-consult" in response
    assert "[THOUGHT][归类机器人]" in response
    session = agent.session_registry.get_or_create("s1")
    assert session.state.value == StateEnum.CONSULT


def test_unrelated_request_gets_standard_rejection(monkeypatch, tmp_path):
    """无关请求：分类为 other → 标准拒绝话术，状态回 CLASSIFY"""
    agent = _make_agent(monkeypatch, tmp_path, ["other"])

    response = asyncio.run(_classify(agent, "今天天气怎么样？", sid="s2"))

    assert "暂不支持该类型任务" in response
    assert "健康理疗、预约相关" in response
    session = agent.session_registry.get_or_create("s2")
    assert session.state.value == StateEnum.CLASSIFY


def test_stream_mode_yields_thought_and_reply(monkeypatch, tmp_path):
    """流式分类：逐 token 产出思考提示与业务回复"""
    agent = _make_agent(monkeypatch, tmp_path, ["query"])

    tokens = asyncio.run(_collect_stream(agent, "肩颈理疗多少钱？", sid="s3"))

    joined = "".join(tokens)
    assert "stub-consult" in joined
    assert any(t.startswith("[THOUGHT][归类机器人]") for t in tokens)


def test_appointment_flow_state_pins_followup_routing(monkeypatch, tmp_path):
    """状态机优先：预约流程中的后续输入不再走 LLM 分类，直接进预约节点"""
    agent = _make_agent(monkeypatch, tmp_path, ["appointment", "query"])

    first = asyncio.run(_classify(agent, "我要预约理疗", sid="s4"))
    assert "stub-appointment" in first

    session = agent.session_registry.get_or_create("s4")
    assert session.state.value == StateEnum.APPOINTMENT

    # 第二轮：即便分类器本会返回 query，状态机也应继续预约流程
    second = asyncio.run(_classify(agent, "明天下午2点", sid="s4"))
    assert "stub-appointment" in second
    assert "stub-consult" not in second


def test_reset_conversation_restores_classify_state(monkeypatch, tmp_path):
    """重置对话：状态回 CLASSIFY，无关回复轮换索引归零"""
    agent = _make_agent(monkeypatch, tmp_path, ["appointment"])
    asyncio.run(_classify(agent, "预约", sid="s5"))

    session = agent.session_registry.get_or_create("s5")
    session.unrelated_reply_index = 3
    assert session.state.value == StateEnum.APPOINTMENT

    agent.reset_conversation("s5")

    assert session.state.value == StateEnum.CLASSIFY
    assert session.unrelated_reply_index == 0


def test_invalid_inputs_rejected_without_crash(monkeypatch, tmp_path):
    """空/无效输入：不崩溃，按无关请求拒绝"""
    agent = _make_agent(monkeypatch, tmp_path, ["other", "other", "other"])

    for i, invalid in enumerate(["", "   ", "？"]):
        response = asyncio.run(_classify(agent, invalid, sid=f"bad{i}"))
        assert isinstance(response, str)
        assert "暂不支持该类型任务" in response


def test_ambiguous_input_routes_without_crash(monkeypatch, tmp_path):
    """模糊输入：分类结果无论落到哪条路由都应产出响应、不崩溃"""
    agent = _make_agent(monkeypatch, tmp_path, ["appointment", "query", "memory"], with_behavior=True)

    for i, text in enumerate(["理疗", "理疗师", "我上次预约的是哪位理疗师"]):
        response = asyncio.run(_classify(agent, text, sid=f"amb{i}"))
        assert len(response) > 0
        assert not response.startswith("[ERROR]处理任务时发生错误")


def test_unknown_category_falls_back_to_rejection(monkeypatch, tmp_path):
    """分类结果不属于四类：无匹配路由 → 标准拒绝话术"""
    agent = _make_agent(monkeypatch, tmp_path, ["不存在的类别"])

    response = asyncio.run(_classify(agent, "随便说点啥", sid="s6"))

    assert "暂不支持该类型任务" in response


def test_unrelated_handler_callback_returns_reclassification(monkeypatch, tmp_path):
    """预约 Agent 的转交回调：handle_unrelated 对同一输入重新分类"""
    agent = _make_agent(monkeypatch, tmp_path, ["query"])
    session = agent.session_registry.get_or_create("s7")

    result = asyncio.run(agent.handle_unrelated("理疗有什么项目？", session))

    assert "stub-consult" in result


async def _collect_stream(agent, text, sid):
    tokens = []
    async for t in agent.classify_task_stream(text, session_id=sid):
        tokens.append(t)
    return tokens
