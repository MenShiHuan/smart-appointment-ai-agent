"""
聊天处理门面（API 层）

薄门面：全局 Agent 单例、图构建/关闭、会话恢复与流式管线的实现均位于
application/agent_runtime.py（Web&Application 层组装根）；本模块保留
图状态缓存（_graph/_checkpointer/_graph_lock）并逐行委托。

注：图状态必须留在本模块——离线测试通过 monkeypatch 本模块 _graph
注入 stub 图，模块 setattr 直写字典不经过任何钩子。
"""
import asyncio

from application.agent_runtime import (
    build_runtime_graph,
    close_runtime_graph,
    stream_user_input,
    task_agent,
)

# LangGraph 编排图（B5：/chat 请求走图编排路径）
# checkpointer 需在事件循环内构造（AsyncSqliteSaver 要求 running loop），
# 因此图在首次请求时懒加载构建；并发首请求用锁保护。
_graph = None
_graph_lock = asyncio.Lock()
_checkpointer = None


async def _get_graph():
    """获取（懒加载）LangGraph 编排图

    首次构建必须在事件循环内创建 SQLite checkpointer，因此延迟到首个请求执行；
    双重检查 + 锁：并发首请求也不会重复构建图与 checkpointer。
    """
    global _graph, _checkpointer
    if _graph is None:
        async with _graph_lock:
            # 双重检查：抢到锁后再判空一次，避免并发重复构建
            if _graph is None:
                _checkpointer, _graph = await build_runtime_graph(task_agent)
    return _graph


async def close_graph_checkpointer():
    """关闭 checkpointer 的 aiosqlite 连接（进程退出前调用，避免挂起）"""
    global _checkpointer, _graph
    _checkpointer, _graph = await close_runtime_graph(_checkpointer, _graph)


def new_session_id() -> str:
    """生成新的会话 ID（前端接入多会话时调用）"""
    return task_agent.session_registry.new_session_id()


async def ProcessUserInput_stream(user_input, session_id=None):
    """
    user_input: 用户输入
    session_id: 会话 ID（多会话隔离的关键；为空时自动新建会话）

    请求经 LangGraph 图编排处理：意图分类 → 业务节点路由，
    档期冲突自愈回环、落库前确认、禁忌筛查均在图中生效。

    异步生成器：逐 token yield 给 /chat/stream 透传；每轮对话经
    TraceContext 记录耗时与 token 数（写 logs/traces.jsonl 并聚合到 /metrics）。
    """
    if not session_id:
        session_id = task_agent.session_registry.new_session_id()

    graph = await _get_graph()
    async for token in stream_user_input(graph, task_agent, user_input, session_id):
        yield token


__all__ = ["task_agent", "ProcessUserInput_stream", "close_graph_checkpointer", "new_session_id"]
