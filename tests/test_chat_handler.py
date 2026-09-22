"""
API 层切换测试（B5 验收）

验证 ProcessUserInput_stream 走 LangGraph 图编排路径：
1. 流式 token 正常透出
2. 未传 session_id 时自动新建会话
全部离线（用 stub 图替换 chat_handler 的全局图）。
"""
import asyncio
import pytest

from agents.core.session import SessionRegistry
from agents.graph import build_agent_graph


class StubClassifier:
    async def classify_task(self, task):
        return "query"


class StubConsultant:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def consult_stream(self, user_input, session=None):
        yield "[REPLY][咨询机器人]graph-stub-response"


class StubAppointment:
    async def run_stream(self, user_input=None, session=None,
                         enable_replan_signal=False, enable_confirm_signal=False):
        yield "[REPLY][预约机器人]stub"


class StubTaskAgent:
    def __init__(self, db_url):
        self.task_classifier = StubClassifier()
        self.appointment_agent = StubAppointment()
        self.consultant_agent = StubConsultant()
        self.behavior_agent = None
        self.session_registry = SessionRegistry(db_path=db_url)


async def _collect(user_input, session_id):
    from api.chat_handler import ProcessUserInput_stream
    tokens = []
    async for token in ProcessUserInput_stream(user_input, session_id=session_id):
        tokens.append(token)
    return "".join(tokens)


def test_process_user_input_streams_via_graph(tmp_path, monkeypatch):
    """请求经 LangGraph 图处理，token 正常透出"""
    import api.chat_handler as handler

    stub_graph = build_agent_graph(StubTaskAgent(f"sqlite:///{tmp_path}/g.db"))
    monkeypatch.setattr(handler, "_graph", stub_graph)

    joined = asyncio.run(_collect("理疗有什么项目？", "s1"))

    assert "graph-stub-response" in joined
    # 图 classify 节点透出思考提示
    assert "[THOUGHT][归类机器人]" in joined


def test_missing_session_id_creates_new_session(tmp_path, monkeypatch):
    """未传 session_id 时自动新建会话，不报错"""
    import api.chat_handler as handler

    stub_graph = build_agent_graph(StubTaskAgent(f"sqlite:///{tmp_path}/g.db"))
    monkeypatch.setattr(handler, "_graph", stub_graph)

    joined = asyncio.run(_collect("你好", None))

    assert "graph-stub-response" in joined
