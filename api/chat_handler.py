import asyncio
import os

from agents.task_classification_agent import TaskClassificationAgent
from agents.appointment_agent import AppointmentAgent
from agents.consultant_agent import ConsultantAgent
from agents.user_behavior_agent import UserBehaviorAgent
from config.constants import StateEnum

# 无状态单例 Agent：所有会话状态按 session_id 隔离在 SessionRegistry 中
# 四个智能体：意图分类（主控）、智能预约、知识咨询、长记忆偏好总结
task_agent = TaskClassificationAgent(
    AppointmentAgent(),
    ConsultantAgent(),
    UserBehaviorAgent()
)

# LangGraph 编排图（B5：/chat 请求走图编排路径）
# checkpointer 需在事件循环内构造（AsyncSqliteSaver 要求 running loop），
# 因此图在首次请求时懒加载构建；并发首请求用锁保护。
_graph = None
_graph_lock = asyncio.Lock()
_checkpointer = None


async def _get_graph():
    global _graph, _checkpointer
    if _graph is None:
        async with _graph_lock:
            if _graph is None:
                from agents.graph import build_agent_graph, make_sqlite_checkpointer
                # CHECKPOINT_DB_PATH 供评测隔离注入
                db_path = os.environ.get("CHECKPOINT_DB_PATH", "data/checkpoints.db")
                _checkpointer = await make_sqlite_checkpointer(db_path)
                _graph = build_agent_graph(task_agent, checkpointer=_checkpointer)
    return _graph


async def close_graph_checkpointer():
    """关闭 checkpointer 的 aiosqlite 连接（进程退出前调用，避免挂起）"""
    global _checkpointer, _graph
    if _checkpointer is not None:
        try:
            await _checkpointer.aconn.close()
        except Exception:
            pass
        _checkpointer = None
        _graph = None


def new_session_id() -> str:
    """生成新的会话 ID（前端接入多会话时调用）"""
    return task_agent.session_registry.new_session_id()


async def ProcessUserInput_stream(user_input, session_id=None):
    """
    user_input: 用户输入
    session_id: 会话 ID（多会话隔离的关键；为空时自动新建会话）

    请求经 LangGraph 图编排处理：意图分类 → 业务节点路由，
    档期冲突自愈回环、落库前确认、禁忌筛查均在图中生效。
    """
    from agents.graph import stream_graph_task

    if not session_id:
        session_id = task_agent.session_registry.new_session_id()

    graph = await _get_graph()
    config = {"configurable": {"thread_id": session_id}}

    from services.trace import TraceContext
    trace = TraceContext("graph_turn", session_id)

    # C12：从 checkpoint 恢复预约槽位与会话状态（服务重启后会话可继续）
    try:
        checkpointer = getattr(graph, "checkpointer", None)
        if checkpointer is not None:
            snapshot = await checkpointer.aget(config)
            values = snapshot.values if snapshot else {}
            if values.get("appointment_history") or values.get("session_state"):
                session = task_agent.session_registry.get_or_create(session_id)
                if values.get("appointment_history"):
                    session.appointment_history = values["appointment_history"]
                if values.get("session_state"):
                    try:
                        session.state.value = StateEnum(values["session_state"])
                    except ValueError:
                        session.state.value = StateEnum.CLASSIFY
    except Exception:
        # 恢复失败不阻塞本轮处理
        pass

    with trace:
        async for token in stream_graph_task(graph, user_input, session_id, config):
            trace.count_tokens()
            yield token
