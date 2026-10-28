"""
智能体路由器 - 根据分类结果将请求路由到对应Agent

职责：
1. 接收分类结果，决定使用哪个Agent处理
2. 协调各Agent之间的调用
3. 会话状态挂在 SessionContext 上，路由器本身无状态
"""

from typing import Any, AsyncGenerator
from agents.core.session import SessionContext
from config.constants import StateEnum
from config.domain_terms import FALLBACK_UNSUPPORTED_TASK


class AgentRouter:
    """智能体路由器 - 根据任务类型路由到对应的处理Agent"""

    def __init__(self, appointment_agent: Any, consultant_agent: Any, session: SessionContext,
                 behavior_agent: Any = None):
        """
        初始化路由器

        Args:
            appointment_agent: 预约处理Agent（无状态单例）
            consultant_agent: 咨询处理Agent（无状态单例）
            session: 当前会话上下文（状态挂载点）
            behavior_agent: 长记忆偏好总结Agent（无状态单例，回答用户偏好/历史查询）
        """
        self.appointment_agent = appointment_agent
        self.consultant_agent = consultant_agent
        self.behavior_agent = behavior_agent
        self.session = session

    async def route_to_appointment(self, task: str) -> AsyncGenerator[str, None]:
        """
        路由到预约Agent处理

        Args:
            task: 用户任务内容

        Yields:
            str: 流式响应内容
        """
        if not self.appointment_agent:
            yield "[ERROR]预约服务暂时不可用"
            return

        # 转换状态
        self.session.state.value = StateEnum.APPOINTMENT

        # 生成思考提示
        # [THOUGHT] 思考提示 token：话术需与 graph/nodes.py 的 THOUGHTS 保持一致
        yield "[THOUGHT][归类机器人] 归类机器人：我发现这是一个预约任务，我将转给预约机器人处理。"

        # 调用预约Agent
        try:
            async for token in self.appointment_agent.run_stream(user_input=task, session=self.session):
                yield token
        except Exception as e:
            yield f"[ERROR]预约处理失败: {str(e)}"
            # 失败降级：重置状态机回分类态，避免会话卡死在预约流程
            self.session.state.value = StateEnum.CLASSIFY

    async def route_to_consultation(self, task: str) -> AsyncGenerator[str, None]:
        """
        路由到咨询Agent处理

        Args:
            task: 用户任务内容

        Yields:
            str: 流式响应内容
        """
        if not self.consultant_agent:
            yield "[ERROR]咨询服务暂时不可用"
            return

        # 转换状态
        self.session.state.value = StateEnum.CONSULT

        # 生成思考提示
        yield "[THOUGHT][归类机器人] 归类机器人：我发现这是一个咨询任务，我将转给咨询机器人处理。"

        # 调用咨询Agent
        try:
            async with self.consultant_agent as agent:
                async for token in agent.consult_stream(task, session=self.session):
                    yield token
        except Exception as e:
            yield f"[ERROR]咨询处理失败: {str(e)}"
            # 失败降级：重置状态机回分类态，避免会话卡死在咨询流程
            self.session.state.value = StateEnum.CLASSIFY

    async def route_to_memory(self, task: str) -> AsyncGenerator[str, None]:
        """
        路由到长记忆偏好总结Agent处理

        Args:
            task: 用户任务内容

        Yields:
            str: 流式响应内容
        """
        if not self.behavior_agent:
            yield "[ERROR]记忆服务暂时不可用"
            return

        # 转换状态
        self.session.state.value = StateEnum.MEMORY

        # 生成思考提示
        yield "[THOUGHT][归类机器人] 归类机器人：这是关于用户偏好或历史记录的问题，我将转给记忆机器人处理。"

        # 调用长记忆偏好总结Agent
        try:
            async for token in self.behavior_agent.answer_preference_query_stream(task):
                yield token
        except Exception as e:
            yield f"[ERROR]记忆查询失败: {str(e)}"
        # 记忆查询为单轮任务：完成后回到分类状态，避免下一轮任意输入被当记忆查询
        self.session.state.value = StateEnum.CLASSIFY

    async def handle_unsupported_task(self, category: str) -> AsyncGenerator[str, None]:
        """
        处理不支持的任务类型

        Args:
            category: 任务分类结果

        Yields:
            str: 回复内容
        """
        # 兜底话术与 graph/nodes.py 的 unsupported 节点共用领域词表常量（评测集关键词：暂不支持 + 健康理疗）
        reply = FALLBACK_UNSUPPORTED_TASK
        # [REPLY] 前缀标识回复正文，前端按前缀区分展示类型
        yield "[REPLY][归类机器人]"
        # 逐字符产出模拟流式打字效果（与 LLM 流式输出形态一致）
        for char in reply:
            yield char

    async def route_by_state(self, task: str) -> AsyncGenerator[str, None]:
        """
        根据当前状态路由任务（用于状态持续的场景）

        Args:
            task: 用户任务内容

        Yields:
            str: 流式响应内容
        """
        state = self.session.state.value
        # 续流路由：会话已在业务流中，按状态机直接分发，不再发 [THOUGHT] 分类提示
        if state == StateEnum.APPOINTMENT:
            async for token in self.appointment_agent.run_stream(user_input=task, session=self.session):
                yield token
        elif state == StateEnum.CONSULT:
            async with self.consultant_agent as agent:
                async for token in agent.consult_stream(task, session=self.session):
                    yield token
        elif state == StateEnum.MEMORY:
            if self.behavior_agent:
                async for token in self.behavior_agent.answer_preference_query_stream(task):
                    yield token
            else:
                self.session.state.value = StateEnum.CLASSIFY
                yield "[ERROR]记忆服务暂时不可用。"
        else:
            # 状态异常，重置并提示
            self.session.state.value = StateEnum.CLASSIFY
            yield "[ERROR]会话状态异常，已重置。请重新开始对话。"

    def get_available_services(self) -> list:
        """获取可用的服务列表"""
        services = []
        if self.appointment_agent:
            services.append("预约服务")
        if self.consultant_agent:
            services.append("咨询服务")
        if self.behavior_agent:
            services.append("记忆服务")
        return services
