"""
会话上下文与多会话隔离

职责：
1. SessionContext：单会话的全部可变状态（对话状态机、预约槽位、聊天历史）
2. PersistentChatHistory：带持久化的聊天历史（内存 + SQLite 双写）
3. SessionRegistry：按 session_id 键控的会话注册表，支持会话恢复

设计要点：
- 消灭全局 SharedState / global_session_id / default_user 单会话假设
- 各 Agent 实例变为无状态单例，所有可变状态挂到 SessionContext 上
- 消息写入聊天历史时自动落库，服务重启后可恢复会话
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from config.constants import SharedState

DEFAULT_DB_PATH = 'sqlite:///data/smart_appointment.db'


def new_appointment_history() -> Dict[str, Any]:
    """创建空的预约槽位状态"""
    return {
        "gender": None,
        "start_time": None,
        "duration": None,
        "project": None,
        "preference": None,
        "technician": None,
        "technician_name": None,
    }


class PersistentChatHistory(InMemoryChatMessageHistory):
    """带持久化的聊天历史：内存操作 + SQLite 落库"""

    def __init__(self, session_id: str, repo):
        super().__init__()
        self._session_id = session_id
        self._repo = repo
        self._loading = False

    def load(self) -> None:
        """从数据库恢复历史消息（仅用于新建上下文时）"""
        self._loading = True
        try:
            for msg in self._repo.get_messages(self._session_id):
                if msg["role"] == "user":
                    super().add_message(HumanMessage(content=msg["content"]))
                else:
                    super().add_message(AIMessage(content=msg["content"]))
        finally:
            self._loading = False

    def add_message(self, message) -> None:
        """添加消息：先入内存，再落库"""
        super().add_message(message)
        if not self._loading:
            role = "user" if message.type == "human" else "assistant"
            self._repo.add_message(self._session_id, role, message.content)

    def clear(self) -> None:
        """清空历史：内存与数据库同步清除"""
        super().clear()
        if not self._loading:
            self._repo.delete_session_messages(self._session_id)


@dataclass
class SessionContext:
    """单会话的完整可变状态"""
    session_id: str
    state: SharedState = field(default_factory=SharedState)
    appointment_history: Dict[str, Any] = field(default_factory=new_appointment_history)
    chat_history: InMemoryChatMessageHistory = field(default_factory=InMemoryChatMessageHistory)
    unrelated_reply_index: int = 0
    updated_at: float = field(default_factory=time.time)


class SessionRegistry:
    """按 session_id 键控的会话注册表"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        # 会话消息持久化经 Services 层访问，遵守五层架构
        from services.conversation_service import ConversationService
        self._conversation_service = ConversationService(db_path)
        self._sessions: Dict[str, SessionContext] = {}

    @property
    def conversation_repo(self):
        return self._conversation_service.conversation_repo

    def new_session_id(self) -> str:
        """生成新的会话 ID"""
        return str(uuid.uuid4())

    def get_or_create(self, session_id: Optional[str]) -> SessionContext:
        """获取或创建会话上下文；已存在则恢复历史消息"""
        if not session_id:
            session_id = self.new_session_id()
        ctx = self._sessions.get(session_id)
        if ctx is None:
            history = PersistentChatHistory(session_id, self.conversation_repo)
            history.load()
            ctx = SessionContext(session_id=session_id, chat_history=history)
            self._sessions[session_id] = ctx
        ctx.updated_at = time.time()
        return ctx

    def clear_session(self, session_id: str) -> None:
        """清空并移除会话（内存与数据库）"""
        ctx = self._sessions.pop(session_id, None)
        if ctx is not None:
            ctx.chat_history.clear()
