"""
无关请求处理器 - 专门负责处理与业务无关的用户请求

职责：
1. 识别和处理与业务无关的请求
2. 提供友好的拒绝回复
3. 引导用户回到正确的业务轨道
4. 回复轮换索引挂载在会话上（多会话隔离）
"""

from typing import AsyncGenerator, Optional
from .state_manager import StateManager
from agents.core.session import SessionContext


class UnrelatedHandler:
    """无关请求处理器 - 处理与业务无关的用户请求"""

    def __init__(self, state_manager: StateManager, session: Optional[SessionContext] = None,
                 business_context: str = "健康理疗服务"):
        """
        初始化无关请求处理器

        Args:
            state_manager: 状态管理器（包装当前会话状态）
            session: 当前会话上下文（回复轮换索引的挂载点）
            business_context: 业务场景名称，用于生成拒绝话术
        """
        self.state_manager = state_manager
        self.session = session
        self._business_context = business_context
        self._default_replies = self._build_replies(business_context)

    def _build_replies(self, service_name: str) -> list:
        """按业务上下文生成拒绝话术模板"""
        return [
            f"抱歉，我无法处理这个问题。我只能帮您处理{service_name}相关的咨询和预约。请问您需要了解我们的服务项目或者预约服务吗？",
            f"很抱歉，我专门负责{service_name}相关的服务。如果您想了解我们的服务内容或进行预约，我很乐意为您提供帮助！",
            f"对不起，这个问题超出了我的服务范围。我主要协助处理{service_name}和相关咨询，有什么可以为您服务的吗？"
        ]

    async def handle_unrelated_sync(self, user_input: str) -> str:
        """
        同步处理无关请求（返回字符串）

        Args:
            user_input: 用户输入内容

        Returns:
            str: 处理结果
        """
        print("归类机器人接管处理 unrelated user_input")

        # 重置状态为分类状态，准备处理下一个输入
        self.state_manager.reset_to_classify()

        # 返回友好的拒绝回复
        return self._get_next_reply()

    async def handle_unrelated_async(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        异步处理无关请求（返回流式响应）

        Args:
            user_input: 用户输入内容

        Yields:
            str: 流式响应内容
        """
        print("归类机器人接管处理 unrelated user_input (async stream)")

        # 重置状态为分类状态
        self.state_manager.reset_to_classify()

        # 生成流式回复
        reply = self._get_next_reply()
        yield "[REPLY][归类机器人]"
        for char in reply:
            yield char

    def _get_next_reply(self) -> str:
        """获取下一个回复内容（轮换索引按会话隔离）"""
        if self.session is None:
            return self._default_replies[0]
        reply = self._default_replies[self.session.unrelated_reply_index % len(self._default_replies)]
        self.session.unrelated_reply_index += 1
        return reply

    def add_custom_reply(self, reply: str) -> None:
        """添加自定义回复"""
        if reply and reply not in self._default_replies:
            self._default_replies.append(reply)

    def set_business_context(self, service_name: str = "健康理疗服务") -> None:
        """设置业务上下文，自定义回复中的服务名称"""
        self._business_context = service_name
        self._default_replies = self._build_replies(service_name)

    def get_available_replies(self) -> list:
        """获取所有可用的回复模板"""
        return self._default_replies.copy()

    def reset_reply_rotation(self) -> None:
        """重置回复轮换索引（会话级）"""
        if self.session is not None:
            self.session.unrelated_reply_index = 0
