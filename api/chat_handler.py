from agents.task_classification_agent import TaskClassificationAgent
from agents.appointment_agent import AppointmentAgent
from agents.consultant_agent import ConsultantAgent
from agents.user_behavior_agent import UserBehaviorAgent

# 无状态单例 Agent：所有会话状态按 session_id 隔离在 SessionRegistry 中
# 四个智能体：意图分类（主控）、智能预约、知识咨询、长记忆偏好总结
task_agent = TaskClassificationAgent(
    AppointmentAgent(),
    ConsultantAgent(),
    UserBehaviorAgent()
)

# LangGraph 编排图（B5：/chat 请求走图编排路径）
# checkpointer 需在事件循环内构造（AsyncSqliteSaver 要求 running loop），
# 因此图在首次请求时懒加载构建。
_graph = None


async def _get_graph():
    global _graph
    if _graph is None:
        from agents.graph import build_agent_graph, make_sqlite_checkpointer
        checkpointer = await make_sqlite_checkpointer("data/checkpoints.db")
        _graph = build_agent_graph(task_agent, checkpointer=checkpointer)
    return _graph


def new_session_id() -> str:
    """生成新的会话 ID（前端接入多会话时调用）"""
    return task_agent.session_registry.new_session_id()


async def ProcessUserInput_stream(user_input, session_id=None, state=None, context=None):
    """
    user_input: 用户输入
    session_id: 会话 ID（多会话隔离的关键；为空时自动新建会话）
    state/context: 兼容旧接口参数（已废弃，状态由 LangGraph 图 + 会话上下文管理）

    请求经 LangGraph 图编排处理：意图分类 → 业务节点路由，
    档期冲突自愈回环、落库前确认、禁忌筛查均在图中生效。
    """
    from agents.graph import stream_graph_task

    if not session_id:
        session_id = task_agent.session_registry.new_session_id()

    graph = await _get_graph()
    async for token in stream_graph_task(graph, user_input, session_id):
        yield token
