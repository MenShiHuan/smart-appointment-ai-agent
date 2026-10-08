import uuid
from langchain_core.chat_history import InMemoryChatMessageHistory
from config.model_provider import create_chat_model
from config.constants import StateEnum
from agents.core.session import SessionContext
from .consultant import (
    KnowledgeRetriever,
    ConsultationClassifier,
    ResponseGenerator,
    ConsultationProcessor
)


class ConsultantAgent:
    """
    咨询机器人主控制器

    职责：
    1. 初始化各个组件
    2. 协调整个咨询流程（实例本身无状态，会话状态挂载在 SessionContext 上）
    """

    def __init__(self, session_id=None, memory_recaller=None):
        # 基础设置（default_session_id 仅用于无会话的独立运行场景）
        self.default_session_id = session_id or str(uuid.uuid4())
        self.unrelated_callback = None

        # 初始化LLM
        self.llm = self._initialize_llm()

        # 初始化组件
        self.knowledge_retriever = KnowledgeRetriever()
        self.consultation_classifier = ConsultationClassifier(self.llm)
        self.response_generator = ResponseGenerator(self.llm)
        # C4b：未显式注入时使用默认记忆召回器（MemoryService.recall）
        if memory_recaller is None:
            memory_recaller = self._default_memory_recaller()
        self.consultation_processor = ConsultationProcessor(
            self.knowledge_retriever,
            self.consultation_classifier,
            self.response_generator,
            memory_recaller=memory_recaller,
        )

    @staticmethod
    def _default_memory_recaller():
        """默认召回器：基于分层记忆条目加权召回（单用户 default_user）"""
        from services.memory_service import MemoryService
        service = MemoryService()

        async def recaller(user_input):
            entries = await service.recall("default_user", user_input)
            return [{"entry_type": e["entry_type"], "content": e["content"]}
                    for e in entries]

        return recaller

    def _initialize_llm(self):
        """初始化通用聊天模型"""
        return create_chat_model(temperature=0.3)

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.knowledge_retriever.initialize()
        print("咨询机器人已启动（数据库RAG模式）")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """异步上下文管理器出口"""
        pass

    def _transient_session(self) -> SessionContext:
        """无会话场景下的临时上下文（独立运行/向后兼容）"""
        return SessionContext(
            session_id=self.default_session_id,
            chat_history=InMemoryChatMessageHistory()
        )

    def set_unrelated_callback(self, callback):
        """设置处理非相关任务的回调函数"""
        self.unrelated_callback = callback

    async def consult(self, user_input: str) -> str:
        """
        基础咨询功能

        用于非流式的简单咨询场景
        """
        return await self.consultation_processor.process_consultation(user_input)

    async def consult_stream(self, user_input: str, session: SessionContext = None):
        """
        流式输出咨询结果

        这是主要的咨询入口点，协调各个组件完成咨询流程。
        会话状态（状态机）取自 session 参数。
        """
        ctx = session or self._transient_session()

        # 1. 检查是否与咨询相关
        is_consultation = await self.consultation_classifier.is_consultation_related(user_input)

        if not is_consultation:
            # 2. 处理与咨询无关的请求
            async for token in self.consultation_processor.handle_unrelated_request(
                user_input, self.unrelated_callback, ctx
            ):
                yield token
            return

        # 3. 处理咨询相关的请求
        async for token in self.consultation_processor.process_consultation_stream(
            user_input, ctx.session_id
        ):
            yield token

        # 4. 重置状态
        ctx.state.value = StateEnum.CLASSIFY
