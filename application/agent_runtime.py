"""
应用组装根（Web&Application 层）

职责：
1. 全局 Agent 单例组装（task_agent）
2. LangGraph 图与 checkpointer 的构建/关闭（事件循环内懒加载由调用方控制）
3. 会话恢复（checkpoint → SessionRegistry）
4. 流式管线：TraceContext 计时 → stream_graph_task 透传

依赖方向：application → agents / services / config（自上而下），不 import api 或 web。
"""
import os
from typing import AsyncIterator


def build_task_agent():
    """组装四个智能体单例：意图分类（主控）、智能预约、知识咨询、长记忆偏好总结"""
    # 组装根自加载 .env：api 层/测试直接 import 本包时，环境变量必须在建模型前就位
    # （服务入口 app.py 已先加载；load_dotenv 不覆盖既有变量，重复调用无害）
    from dotenv import load_dotenv
    load_dotenv()

    from agents.task_classification_agent import TaskClassificationAgent
    from agents.appointment_agent import AppointmentAgent
    from agents.consultant_agent import ConsultantAgent
    from agents.user_behavior_agent import UserBehaviorAgent
    return TaskClassificationAgent(
        AppointmentAgent(),
        ConsultantAgent(),
        UserBehaviorAgent()
    )


# 全局单例：无状态 Agent，所有会话状态按 session_id 隔离在 SessionRegistry 中
task_agent = build_task_agent()


async def build_runtime_graph(agent):
    """构建 LangGraph 图 + SQLite checkpointer（必须在事件循环内调用）

    返回 (checkpointer, graph)。CHECKPOINT_DB_PATH 供评测隔离注入。
    """
    from agents.graph import build_agent_graph, make_sqlite_checkpointer
    db_path = os.environ.get("CHECKPOINT_DB_PATH", "data/checkpoints.db")
    checkpointer = await make_sqlite_checkpointer(db_path)
    graph = build_agent_graph(agent, checkpointer=checkpointer)
    return checkpointer, graph


async def close_runtime_graph(checkpointer, graph):
    """关闭 checkpointer 的 aiosqlite 连接（进程退出前调用，避免挂起）

    返回 (None, None)：由调用方（门面）回写全局状态。
    """
    if checkpointer is not None:
        try:
            await checkpointer.aconn.close()
        except Exception:
            pass
    return None, None


async def restore_session_from_checkpoint(graph, agent, session_id) -> None:
    """从 checkpoint 恢复预约槽位与会话状态（服务重启后会话可继续）"""
    from config.constants import StateEnum

    try:
        checkpointer = getattr(graph, "checkpointer", None)
        if checkpointer is not None:
            config = {"configurable": {"thread_id": session_id}}
            snapshot = await checkpointer.aget(config)
            values = snapshot.values if snapshot else {}
            if values.get("appointment_history") or values.get("session_state"):
                session = agent.session_registry.get_or_create(session_id)
                if values.get("appointment_history"):
                    session.appointment_history = values["appointment_history"]
                if values.get("session_state"):
                    try:
                        session.state.value = StateEnum(values["session_state"])
                    except ValueError:
                        # 存档中的状态值非法（枚举外值）时回退到初始分类状态，不阻塞本轮对话
                        session.state.value = StateEnum.CLASSIFY
    except Exception:
        # 恢复失败不阻塞本轮处理
        pass


async def stream_user_input(graph, agent, user_input, session_id) -> AsyncIterator[str]:
    """流式管线：会话恢复 → TraceContext 计时 → stream_graph_task 透传"""
    from agents.graph import stream_graph_task
    from services.trace import TraceContext

    config = {"configurable": {"thread_id": session_id}}
    await restore_session_from_checkpoint(graph, agent, session_id)

    # 每轮对话一个 TraceContext：统计本轮阶段耗时与 token 数，供 /metrics 与日志观察
    trace = TraceContext("graph_turn", session_id)
    with trace:
        async for token in stream_graph_task(graph, user_input, session_id, config):
            trace.count_tokens()
            yield token


__all__ = [
    "task_agent",
    "build_task_agent",
    "build_runtime_graph",
    "close_runtime_graph",
    "restore_session_from_checkpoint",
    "stream_user_input",
]
