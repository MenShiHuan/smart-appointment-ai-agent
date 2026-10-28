"""
LangGraph 图节点

节点与业务步骤一一对应：意图分类 / 预约 / 咨询 / 记忆 / 兜底 /
档期冲突重规划（replan）/ 落库前确认（confirm）/ 禁忌转人工（manual_review）。

节点通过 get_stream_writer 把 token 实时透出（保持真流式体验），
同时把 token 聚合进 state["reply"] 供 checkpointer 回放与调试。
"""
from langgraph.config import get_stream_writer

from agents.core.session import new_appointment_history
from config.constants import StateEnum
from config.domain_terms import FALLBACK_UNSUPPORTED_TASK
from .state import AgentGraphState

# 分类后的思考提示（与旧 AgentRouter 的输出保持一致）
THOUGHTS = {
    "appointment": "[THOUGHT][归类机器人] 归类机器人：我发现这是一个预约任务，我将转给预约机器人处理。",
    "query": "[THOUGHT][归类机器人] 归类机器人：我发现这是一个咨询任务，我将转给咨询机器人处理。",
    "memory": "[THOUGHT][归类机器人] 归类机器人：这是关于用户偏好或历史记录的问题，我将转给记忆机器人处理。",
}


def _get_session(task_agent, state: AgentGraphState, config):
    """从图状态/配置中解析会话上下文"""
    session_id = state.get("session_id") or (config or {}).get("configurable", {}).get("thread_id")
    return task_agent.session_registry.get_or_create(session_id)


def make_classify_node(task_agent):
    """意图分类节点：会话已处于业务流时按状态路由（免 LLM 分类），否则 LLM 分类"""
    async def classify_node(state: AgentGraphState, config) -> dict:
        session = _get_session(task_agent, state, config)

        # 预约/咨询/记忆流程进行中：按状态机路由，不再调 LLM 分类
        state_value = session.state.value
        if state_value == StateEnum.APPOINTMENT:
            return {"category": "appointment"}
        if state_value == StateEnum.CONSULT:
            return {"category": "query"}
        if state_value == StateEnum.MEMORY:
            return {"category": "memory"}

        category = await task_agent.task_classifier.classify_task(state["user_input"])
        thought = THOUGHTS.get(category, "")
        if thought:
            get_stream_writer()(thought)
        return {"category": category, "session_state": session.state.value}
    return classify_node


def make_appointment_node(task_agent):
    """预约节点：委托预约 Agent 处理本轮输入；捕获冲突/确认/禁忌信号交由对应节点"""
    async def appointment_node(state: AgentGraphState, config) -> dict:
        from .replan import BREAKER_MESSAGE, MAX_REPLAN_ROUNDS
        writer = get_stream_writer()
        session = _get_session(task_agent, state, config)
        session.state.value = StateEnum.APPOINTMENT

        reply = []
        replan_signal = False
        confirmation_signal = False
        manual_review_signal = False
        async for token in task_agent.appointment_agent.run_stream(
            user_input=state["user_input"], session=session,
            enable_replan_signal=True, enable_confirm_signal=True
        ):
            if token == "[SIGNAL]replan_needed":
                replan_signal = True
                continue
            if token == "[SIGNAL]confirmation_pending":
                confirmation_signal = True
                continue
            if token == "[SIGNAL]manual_review":
                manual_review_signal = True
                continue
            if replan_signal or confirmation_signal or manual_review_signal:
                # 抑制信号后的消息，交由后续节点决定最终回复
                continue
            reply.append(token)
            writer(token)

        # C12：槽位与状态机随 checkpointer 持久化（重启后恢复会话）
        session_snapshot = {
            "appointment_history": session.appointment_history,
            "session_state": session.state.value,
        }
        if manual_review_signal:
            return {"reply": reply, "needs_manual_review": True, **session_snapshot}
        if confirmation_signal:
            return {"reply": reply, "needs_confirmation": True, **session_snapshot}
        if replan_signal:
            if state.get("loop_count", 0) >= MAX_REPLAN_ROUNDS:
                # 已达重规划上限：熔断降级
                writer(BREAKER_MESSAGE)
                return {"reply": reply, "needs_replan": False, **session_snapshot}
            return {"reply": reply, "needs_replan": True, **session_snapshot}
        return {"reply": reply, "needs_replan": False, **session_snapshot}
    return appointment_node


def make_consultation_node(task_agent):
    """咨询节点：委托咨询 Agent 处理本轮输入"""
    async def consultation_node(state: AgentGraphState, config) -> dict:
        writer = get_stream_writer()
        session = _get_session(task_agent, state, config)
        session.state.value = StateEnum.CONSULT
        reply = []
        async with task_agent.consultant_agent as agent:
            async for token in agent.consult_stream(state["user_input"], session=session):
                reply.append(token)
                writer(token)
        return {"reply": reply, "session_state": session.state.value}
    return consultation_node


def make_memory_node(task_agent):
    """记忆节点：委托长记忆偏好总结 Agent 回答用户偏好/历史查询"""
    async def memory_node(state: AgentGraphState, config) -> dict:
        writer = get_stream_writer()
        session = _get_session(task_agent, state, config)
        session.state.value = StateEnum.MEMORY
        reply = []
        async for token in task_agent.behavior_agent.answer_preference_query_stream(
            state["user_input"]
        ):
            reply.append(token)
            writer(token)
        # 记忆查询一问一答：回答完回到分类态，下一轮输入重新分类
        session.state.value = StateEnum.CLASSIFY
        return {"reply": reply, "session_state": session.state.value}
    return memory_node


def make_replan_node(task_agent):
    """档期冲突重规划节点：LLM 建议替代时间 → 以改约输入回流预约节点（有界循环 + 熔断）"""
    async def replan_node(state: AgentGraphState, config) -> dict:
        from .replan import ReplanHandler, BREAKER_MESSAGE
        writer = get_stream_writer()
        session = _get_session(task_agent, state, config)

        handler = ReplanHandler(task_agent.appointment_agent.llm)
        suggestion = await handler.suggest_time(session.appointment_history)
        suggested_time = suggestion.get("suggested_time", "")

        technician_name = session.appointment_history.get("technician_name") or ""
        tried_hash = handler.attempt_hash(technician_name, suggested_time)
        tried_hashes = state.get("tried_hashes", [])

        # 护栏：兜底失败或相同尝试重复 → 熔断降级
        if not suggested_time or tried_hash in tried_hashes:
            print(f"[replan] 熔断：suggested={suggested_time!r}, repeated={tried_hash in tried_hashes}")
            writer(BREAKER_MESSAGE)
            return {
                "needs_replan": False,
                "loop_count": state.get("loop_count", 0) + 1,
                "tried_hashes": tried_hashes,
            }

        writer(
            f"[THOUGHT][预约机器人] 原时段已满，尝试为您调整到 {suggested_time}"
            f"（{suggestion.get('reason', '')}）。\n"
        )
        # 模拟用户改约输入，回流预约节点走完整的抽取→匹配→落库流程
        return {
            "needs_replan": True,
            "loop_count": state.get("loop_count", 0) + 1,
            "tried_hashes": tried_hashes + [tried_hash],
            "user_input": f"请帮我改约到 {suggested_time} 这个时间",
        }
    return replan_node


def make_confirm_node(task_agent):
    """落库前确认节点：输出预约摘要与禁忌筛查问询（B4）"""
    async def confirm_node(state: AgentGraphState, config) -> dict:
        writer = get_stream_writer()
        session = _get_session(task_agent, state, config)
        tech = session.appointment_history.get('pending_technician')
        message = task_agent.appointment_agent.message_builder.create_booking_confirmation_message(
            tech or {}, session.appointment_history
        )
        writer(f"[REPLY][预约机器人]{message}")
        return {"session_state": session.state.value}
    return confirm_node


def make_manual_review_node(task_agent):
    """禁忌转人工节点：输出转人工话术并重置会话（B4，不落库）"""
    async def manual_review_node(state: AgentGraphState, config) -> dict:
        writer = get_stream_writer()
        session = _get_session(task_agent, state, config)
        hit_terms = session.appointment_history.get('contraindication_hit', [])
        message = task_agent.appointment_agent.message_builder.create_contraindication_referral_message(hit_terms)
        writer(f"[REPLY][预约机器人]{message}")
        # 转人工：清空预约槽位并重置状态机
        session.appointment_history = new_appointment_history()
        session.state.value = StateEnum.CLASSIFY
        return {
            "appointment_history": session.appointment_history,
            "session_state": session.state.value,
        }
    return manual_review_node


def make_unsupported_node(task_agent):
    """兜底节点：不支持的任务类型"""
    async def unsupported_node(state: AgentGraphState, config) -> dict:
        # 与 agent_router 兜底话术共用领域词表常量（评测集关键词：暂不支持 + 健康理疗）
        reply_text = FALLBACK_UNSUPPORTED_TASK
        tokens = ["[REPLY][归类机器人]", reply_text]
        writer = get_stream_writer()
        for token in tokens:
            writer(token)
        return {"reply": tokens}
    return unsupported_node


def route_by_category(state: AgentGraphState) -> str:
    """分类结果 → 下一节点名"""
    category = state.get("category", "other")
    mapping = {
        "appointment": "appointment",
        "query": "consultation",
        "memory": "memory",
    }
    return mapping.get(category, "unsupported")


def route_after_appointment(state: AgentGraphState) -> str:
    """预约节点后置路由：禁忌转人工 > 落库前确认 > 档期冲突重规划 > 结束"""
    if state.get("needs_manual_review"):
        return "manual_review"
    if state.get("needs_confirmation"):
        return "confirm"
    if state.get("needs_replan"):
        return "replan"
    return "end"


def route_after_replan(state: AgentGraphState) -> str:
    """replan 节点后置路由：继续重试 → 回流预约节点；熔断 → 结束本轮"""
    if state.get("needs_replan"):
        return "retry"
    return "end"
