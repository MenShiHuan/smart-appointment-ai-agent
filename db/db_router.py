"""
数据库路由器

聚合各 Repository 的统一数据访问入口。
"""
from .base import get_session_manager
from .repositories import (TechnicianRepository, KnowledgeRepository,
                           UserBehaviorRepository, ConversationRepository, MemoryRepository,
                           PdfDocumentRepository)


class DatabaseRouter:
    """
    数据库路由器

    职责：
    1. 管理数据库连接和会话
    2. 提供统一的数据访问入口
    3. 协调各个Repository的操作

    会话管理器为进程级共享（按 db_path 键控），db_path 为 None 时
    取 DATABASE_URL 环境变量（评测隔离入口）。
    """

    def __init__(self, db_path: str = None):
        """
        初始化数据库路由器

        Args:
            db_path: 数据库连接路径
        """
        self.session_manager = get_session_manager(db_path)

        # 初始化各个Repository
        self.technician_repo = TechnicianRepository(self.session_manager)
        self.knowledge_repo = KnowledgeRepository(self.session_manager)
        self.user_behavior_repo = UserBehaviorRepository(self.session_manager)
        self.conversation_repo = ConversationRepository(self.session_manager)
        self.memory_repo = MemoryRepository(self.session_manager)
        self.pdf_document_repo = PdfDocumentRepository(self.session_manager)

    @property
    def technicians(self) -> TechnicianRepository:
        """获取技师数据仓库"""
        return self.technician_repo

    @property
    def knowledge(self) -> KnowledgeRepository:
        """获取知识库数据仓库"""
        return self.knowledge_repo

    @property
    def user_behavior(self) -> UserBehaviorRepository:
        """获取用户行为数据仓库"""
        return self.user_behavior_repo

    @property
    def conversation(self) -> ConversationRepository:
        """获取会话消息数据仓库"""
        return self.conversation_repo

    @property
    def memory(self) -> MemoryRepository:
        """获取记忆条目数据仓库"""
        return self.memory_repo

    @property
    def pdf_documents(self) -> PdfDocumentRepository:
        """获取 PDF 文档注册表数据仓库"""
        return self.pdf_document_repo

    def close(self):
        """关闭数据库连接"""
        self.session_manager.close()
