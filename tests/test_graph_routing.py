"""
LangGraph 编排路由测试（B1 验收）

全部离线：stub 分类器 + stub Agent 验证四类意图的路由：
appointment → 预约节点；query → 咨询节点；memory → 记忆节点；other → 兜底节点。
"""
import asyncio
import pytest

from agents.core.session import SessionRegistry
from agents.graph import build_agent_graph, stream_graph_task, nodes, export_mermaid


class StubClassifier:
    def __init__(self, category):
        self.category = category

    async def classify_task(self, task):
        return self.category


class StubAppointmentAgent:
    async def run_stream(self, user_input=None, session=None,
                         enable_replan_signal=False, enable_confirm_signal=False):
        yield "[REPLY][预约机器人]stub-appointment"


class StubConsultantAgent:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def consult_stream(self, user_input, session=None):
        yield "[REPLY][咨询机器人]stub-consultation"


class StubBehaviorAgent:
    async def answer_preference_query_stream(self, user_input, user_id="default_user"):
        yield "[REPLY][记忆机器人]stub-memory"


class StubTaskAgent:
    def __init__(self, category, db_url):
        self.task_classifier = StubClassifier(category)
        self.appointment_agent = StubAppointmentAgent()
        self.consultant_agent = StubConsultantAgent()
        self.behavior_agent = StubBehaviorAgent()
        self.session_registry = SessionRegistry(db_path=db_url)


async def _run_graph(task_agent, session_id):
    graph = build_agent_graph(task_agent)
    tokens = []
    async for token in stream_graph_task(graph, "测试输入", session_id):
        tokens.append(token)
    return tokens


@pytest.mark.parametrize("category,expected_reply", [
    ("appointment", "[REPLY][预约机器人]stub-appointment"),
    ("query", "[REPLY][咨询机器人]stub-consultation"),
    ("memory", "[REPLY][记忆机器人]stub-memory"),
])
def test_routing_by_category(category, expected_reply, tmp_path):
    """三类业务意图路由到对应节点，且带思考提示"""
    task_agent = StubTaskAgent(category, f"sqlite:///{tmp_path}/g.db")
    tokens = asyncio.run(_run_graph(task_agent, "session-1"))

    assert any(expected_reply in t for t in tokens)
    assert any("[THOUGHT][归类机器人]" in t for t in tokens)


def test_unsupported_category_routes_to_fallback(tmp_path):
    """other/pay 等未支持分类 → 兜底节点"""
    task_agent = StubTaskAgent("other", f"sqlite:///{tmp_path}/g.db")
    tokens = asyncio.run(_run_graph(task_agent, "session-1"))

    # 兜底节点逐字符输出，拼接后校验
    joined = "".join(tokens)
    assert "暂不支持该类型任务" in joined


def test_route_by_category_mapping():
    """条件边映射函数（纯函数）"""
    assert nodes.route_by_category({"category": "appointment"}) == "appointment"
    assert nodes.route_by_category({"category": "query"}) == "consultation"
    assert nodes.route_by_category({"category": "memory"}) == "memory"
    assert nodes.route_by_category({"category": "pay"}) == "unsupported"
    assert nodes.route_by_category({"category": "other"}) == "unsupported"
    assert nodes.route_by_category({}) == "unsupported"


def test_graph_exports_mermaid_with_all_nodes(tmp_path):
    """图可编译且 Mermaid 定义包含全部节点"""
    task_agent = StubTaskAgent("query", f"sqlite:///{tmp_path}/g.db")
    graph = build_agent_graph(task_agent)

    mermaid = graph.get_graph().draw_mermaid()

    for node in ("classify", "appointment", "consultation", "memory", "unsupported"):
        assert node in mermaid


def test_session_isolation_across_threads(tmp_path):
    """不同 thread_id 的图运行互不影响会话状态"""
    db_url = f"sqlite:///{tmp_path}/g.db"
    task_agent = StubTaskAgent("query", db_url)

    asyncio.run(_run_graph(task_agent, "session-1"))
    asyncio.run(_run_graph(task_agent, "session-2"))

    assert "session-1" in task_agent.session_registry._sessions
    assert "session-2" in task_agent.session_registry._sessions
