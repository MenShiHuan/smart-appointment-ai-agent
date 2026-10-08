"""
LangGraph 编排图构建器

图结构（B1 骨架）：

    START → classify ──条件边──→ appointment / consultation / memory / unsupported → END

B3 之后 appointment 内部会引入"冲突重规划"回环与确认节点；
B2 接入 SqliteSaver checkpointer（thread_id = session_id）实现持久化。
"""
from langgraph.graph import StateGraph, START, END

from .state import AgentGraphState
from . import nodes


async def make_sqlite_checkpointer(db_path: str = "data/checkpoints.db"):
    """
    创建 SQLite 检查点存储（图状态持久化，thread_id = session_id）

    使用 AsyncSqliteSaver（图节点全部为 async，需要异步 saver；依赖 aiosqlite）。
    注意：
    - 必须在事件循环内构造（AsyncSqliteSaver.__init__ 需要 running loop），
      生产环境在 FastAPI startup 事件中调用；
    - from_conn_string 在 langgraph 1.x 才返回上下文管理器，0.2.x 需直接
      传入 aiosqlite.Connection。
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = aiosqlite.connect(db_path)
    return AsyncSqliteSaver(conn)


def build_agent_graph(task_agent, checkpointer=None):
    """构建意图分类 → 业务节点路由图（可挂 checkpointer 实现持久化/断点恢复）"""
    graph = StateGraph(AgentGraphState)

    graph.add_node("classify", nodes.make_classify_node(task_agent))
    graph.add_node("appointment", nodes.make_appointment_node(task_agent))
    graph.add_node("consultation", nodes.make_consultation_node(task_agent))
    graph.add_node("memory", nodes.make_memory_node(task_agent))
    graph.add_node("unsupported", nodes.make_unsupported_node(task_agent))
    graph.add_node("replan", nodes.make_replan_node(task_agent))
    graph.add_node("confirm", nodes.make_confirm_node(task_agent))
    graph.add_node("manual_review", nodes.make_manual_review_node(task_agent))

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        nodes.route_by_category,
        {
            "appointment": "appointment",
            "consultation": "consultation",
            "memory": "memory",
            "unsupported": "unsupported",
        },
    )
    # 预约节点后置路由：冲突 → replan（有界自愈回环）｜摘要确认 → confirm｜禁忌 → manual_review｜否则结束
    graph.add_conditional_edges(
        "appointment",
        nodes.route_after_appointment,
        {
            "replan": "replan",
            "confirm": "confirm",
            "manual_review": "manual_review",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "replan",
        nodes.route_after_replan,
        {"retry": "appointment", "end": END},
    )
    graph.add_edge("confirm", END)
    graph.add_edge("manual_review", END)
    for target in ("consultation", "memory", "unsupported"):
        graph.add_edge(target, END)

    return graph.compile(checkpointer=checkpointer)


async def stream_graph_task(graph, user_input: str, session_id: str, config: dict = None):
    """
    以图运行一轮任务并流式透出 token

    stream_mode=["custom", "updates"]：custom 承载节点实时写出的 token，
    updates 承载节点状态更新（调试/断点恢复用）。
    """
    if config is None:
        config = {"configurable": {"thread_id": session_id}}
    async for mode, payload in graph.astream(
        {"user_input": user_input, "session_id": session_id},
        config=config,
        stream_mode=["custom", "updates"],
    ):
        if mode == "custom":
            yield payload


def export_mermaid(graph, path: str = "docs/graph.mmd") -> str:
    """导出 Mermaid 图定义（README 架构图素材）"""
    mermaid = graph.get_graph().draw_mermaid()
    with open(path, "w", encoding="utf-8") as f:
        f.write(mermaid)
    return mermaid
