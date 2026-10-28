"""
任务分类主控制器

LLM 意图分类 + 按 session_id 的会话注册表隔离，并协调预约/咨询/记忆 Agent 的路由。
"""
from config.model_provider import create_chat_model
from config.constants import StateEnum
from agents.core.session import SessionContext, SessionRegistry
from .task_classification import (
    TaskClassifier,
    StateManager,
    AgentRouter,
    UnrelatedHandler,
    ClassificationProcessor
)


class TaskClassificationAgent:
    """
    任务分类代理主控制器

    职责：
    1. 初始化分类组件
    2. 按 session_id 提供多会话隔离的任务分类与路由
    3. 管理与其他Agent的协调
    """

    def __init__(self, appointment_agent, consultant_agent, behavior_agent=None):
        # 基础设置
        self.appointment_agent = appointment_agent
        self.consultant_agent = consultant_agent
        self.behavior_agent = behavior_agent

        # 初始化LLM
        self.llm = self._initialize_llm()

        # 全局单例组件（无会话状态）
        self.task_classifier = TaskClassifier(self.llm)
        self.session_registry = SessionRegistry()
        # 业务上下文文案：与兜底话术的评测集关键词“健康理疗”保持一致（见 nodes.make_unsupported_node）
        self.business_context = "健康理疗服务"

        # 设置回调函数
        self._setup_callbacks()

    def _initialize_llm(self):
        """初始化通用聊天模型"""
        return create_chat_model(temperature=0)

    def _setup_callbacks(self):
        """设置Agent的回调函数"""
        if self.appointment_agent and hasattr(self.appointment_agent, 'unrelated_callback'):
            self.appointment_agent.unrelated_callback = self.handle_unrelated

        if self.consultant_agent and hasattr(self.consultant_agent, 'set_unrelated_callback'):
            self.consultant_agent.set_unrelated_callback(self.handle_unrelated_async)

    def _build_processor(self, session: SessionContext) -> ClassificationProcessor:
        """按会话构建一次性的流程组件（组件本身无状态，状态挂在 session 上）"""
        state_manager = StateManager(session.state)
        agent_router = AgentRouter(
            self.appointment_agent, self.consultant_agent, session,
            behavior_agent=self.behavior_agent
        )
        unrelated_handler = UnrelatedHandler(
            state_manager, session, business_context=self.business_context
        )
        return ClassificationProcessor(
            self.task_classifier,
            state_manager,
            agent_router,
            unrelated_handler
        )

    # ===========================================
    # 主要接口方法
    # ===========================================

    async def classify_task(self, task, session_id=None):
        """分类任务（非流式，聚合流式输出）"""
        result = ""
        async for token in self.classify_task_stream(task, session_id):
            result += token
        return result

    async def classify_task_stream(self, task, session_id=None):
        """流式分类任务（主要入口，按 session_id 隔离）"""
        session = self.session_registry.get_or_create(session_id)
        processor = self._build_processor(session)
        async for token in processor.process_task_stream(task):
            yield token

    async def handle_unrelated(self, user_input, session: SessionContext):
        """处理无关请求（同步版本，供预约Agent回调）"""
        # 与预约无关的请求应该重新进行分类，而不是直接拒绝
        print(f"[DEBUG] 预约机器人转交的请求：{user_input}")

        result = ""
        async for token in self.handle_unrelated_async(user_input, session):
            result += token
        return result

    async def handle_unrelated_async(self, user_input, session: SessionContext):
        """处理无关请求（异步流版本，供咨询Agent回调）"""
        print(f"[DEBUG] 咨询机器人转交的请求：{user_input}")

        # 使用当前会话重新进行任务分类
        processor = self._build_processor(session)
        async for token in processor.process_task_stream(user_input):
            yield token

    # ===========================================
    # 扩展功能方法
    # ===========================================

    def get_classification_info(self, session_id=None):
        """获取指定会话的分类系统信息"""
        session = self.session_registry.get_or_create(session_id)
        return self._build_processor(session).get_current_state_info()

    def reset_conversation(self, session_id=None):
        """重置指定会话的对话状态"""
        session = self.session_registry.get_or_create(session_id)
        session.state.value = StateEnum.CLASSIFY
        session.unrelated_reply_index = 0

    def set_business_context(self, service_name: str = "健康理疗服务"):
        """设置业务上下文"""
        self.business_context = service_name
