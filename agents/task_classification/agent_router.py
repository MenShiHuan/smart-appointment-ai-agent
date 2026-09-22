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
        yield "[THOUGHT][归类机器人] 归类机器人：我发现这是一个预约任务，我将转给预约机器人处理。"

        # 调用预约Agent
        try:
            async for token in self.appointment_agent.run_stream(user_input=task, session=self.session):
                yield token
        except Exception as e:
            yield f"[ERROR]预约处理失败: {str(e)}"
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
            self.session.state.value = StateEnum.CLASSIFY

    async def handle_unsupported_task(self, category: str) -> AsyncGenerator[str, None]:
        """
        处理不支持的任务类型

        Args:
            category: 任务分类结果

        Yields:
            str: 回复内容
        """
        reply = "暂不支持该类型任务。请只询问和按摩、预约相关的问题。"
        yield "[REPLY][归类机器人]"
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
