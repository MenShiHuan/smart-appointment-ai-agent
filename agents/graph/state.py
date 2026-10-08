"""
LangGraph 编排状态定义

状态只存可序列化字段（B2 接入 SqliteSaver checkpointer 后需持久化）：
会话内的业务状态（预约槽位、聊天历史）仍挂在 SessionRegistry 的
SessionContext 上，图状态只负责本轮编排流转。
"""
from typing import List, TypedDict


class AgentGraphState(TypedDict, total=False):
    user_input: str      # 本轮用户输入
    session_id: str      # 会话 ID（与 thread_id 一致，节点据此取 SessionContext）
    category: str        # 意图分类结果（appointment/query/memory/other）
    reply: List[str]     # 本轮产出的 token 列表（节点粒度聚合，兜底/回放用）
    loop_count: int      # 自愈循环计数（B3 档期冲突重规划使用）
    needs_replan: bool   # 档期冲突待重规划标记（B3）
    tried_hashes: List[str]  # 已尝试的 (理疗师, 时间) 哈希，用于熔断（B3）
    appointment_history: dict   # 预约槽位快照（C12：随 checkpointer 持久化，重启恢复）
    session_state: str          # 会话状态机状态（C12：随 checkpointer 持久化）
