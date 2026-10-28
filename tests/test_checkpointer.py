"""
Checkpointer 持久化测试（B2 验收）

验证：
1. 图状态写入 SQLite 检查点，新图实例（模拟服务重启）可恢复
2. 不同 thread_id 的检查点互相独立
全部离线。AsyncSqliteSaver 必须在事件循环内构造，测试全程在单个
asyncio.run 内完成（构造 → 运行 → 重启模拟 → 读取）。
"""
import asyncio
import pytest

from agents.core.session import SessionRegistry
from agents.graph import build_agent_graph, stream_graph_task, make_sqlite_checkpointer


class StubClassifier:
    def __init__(self, category):
        self.category = category

    async def classify_task(self, task):
        return self.category


class StubConsultant:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def consult_stream(self, user_input, session=None):
        yield "[REPLY][咨询机器人]stub"


class StubAppointment:
    async def run_stream(self, user_input=None, session=None,
                         enable_replan_signal=False, enable_confirm_signal=False):
        yield "[REPLY][预约机器人]stub"


class StubTaskAgent:
    def __init__(self, category, db_url):
        self.task_classifier = StubClassifier(category)
        self.appointment_agent = StubAppointment()
        self.consultant_agent = StubConsultant()
        self.behavior_agent = None
        self.session_registry = SessionRegistry(db_path=db_url)


async def _scenario(tmp_path, category, sid):
    """运行一轮图任务后模拟重启（同文件新 saver/新图），读取检查点快照"""
    db_url = f"sqlite:///{tmp_path}/g.db"
    ckpt_path = str(tmp_path / "checkpoints.db")

    checkpointer = await make_sqlite_checkpointer(ckpt_path)
    graph = build_agent_graph(
        StubTaskAgent(category, db_url), checkpointer=checkpointer
    )
    try:
        async for _ in stream_graph_task(graph, "测试输入", sid):
            pass

        # 模拟服务重启：同一检查点文件构建新 saver 与新图
        checkpointer2 = await make_sqlite_checkpointer(ckpt_path)
        graph2 = build_agent_graph(
            StubTaskAgent(category, db_url), checkpointer=checkpointer2
        )
        try:
            snapshot = await graph2.aget_state({"configurable": {"thread_id": sid}})
            return snapshot
        finally:
            await checkpointer2.conn.close()
    finally:
        await checkpointer.conn.close()


def test_checkpoint_persists_graph_state(tmp_path):
    """图状态持久化：模拟重启后新图实例可读取历史状态"""
    snapshot = asyncio.run(_scenario(tmp_path, "query", "session-1"))

    assert snapshot is not None
    assert snapshot.values.get("category") == "query"
    assert snapshot.values.get("session_id") == "session-1"


async def _independent_threads_scenario(tmp_path):
    """thread-a 运行后查询未运行过的 thread-b 检查点（应为空）"""
    db_url = f"sqlite:///{tmp_path}/g.db"
    ckpt_path = str(tmp_path / "checkpoints.db")

    checkpointer = await make_sqlite_checkpointer(ckpt_path)
    graph = build_agent_graph(
        StubTaskAgent("appointment", db_url), checkpointer=checkpointer
    )
    try:
        async for _ in stream_graph_task(graph, "测试输入", "thread-a"):
            pass

        # thread-b 从未运行 → 无历史状态
        snapshot_b = await graph.aget_state({"configurable": {"thread_id": "thread-b"}})
        return snapshot_b
    finally:
        await checkpointer.conn.close()


def test_threads_have_independent_checkpoints(tmp_path):
    """不同 thread_id（会话）的检查点互相独立"""
    snapshot_b = asyncio.run(_independent_threads_scenario(tmp_path))

    assert snapshot_b.values == {}
