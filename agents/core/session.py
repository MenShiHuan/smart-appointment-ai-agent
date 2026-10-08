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
    window_summary: Optional[str] = None  # C4b：滚动摘要（窗口压缩产物，挂会话上下文）
    window_summary_upto: int = 0          # C4b：已压缩消息边界（只压缩上次摘要后的新消息）
    updated_at: float = field(default_factory=time.time)


class MessageWindowManager:
    """短期记忆窗口管理器（C4b）

    - 窗口上限：最近 WINDOW_MESSAGE_LIMIT 条消息（默认 20，即 10 轮）
    - 占用达窗口 60% 时触发滚动摘要：压缩最早的消息为摘要（LLM 生成），
      摘要与剩余最近消息拼接继续上下文，控制单会话 Token
    - 历史对象本身不做裁剪（持久化保留全量），窗口只在构建上下文时生效
    """

    def __init__(self, message_limit: Optional[int] = None,
                 trigger_ratio: Optional[float] = None,
                 keep_after_summary: Optional[int] = None):
        from config import memory_config as mc
        self.message_limit = message_limit or mc.WINDOW_MESSAGE_LIMIT
        self.trigger_ratio = trigger_ratio or mc.WINDOW_SUMMARY_TRIGGER_RATIO
        self.keep_after_summary = keep_after_summary or mc.WINDOW_KEEP_AFTER_SUMMARY

    def should_compress(self, messages: list, unsummarized_count: Optional[int] = None) -> bool:
        """未压缩消息数达到窗口 60% 时触发压缩（已压缩过的不重复计入）"""
        count = unsummarized_count if unsummarized_count is not None else len(messages)
        return count >= int(self.message_limit * self.trigger_ratio)

    def split_for_compression(self, messages: list) -> tuple:
        """切分：待压缩的最早消息 + 保留的最近消息"""
        keep_count = min(self.keep_after_summary, len(messages))
        return messages[:-keep_count], messages[-keep_count:]

    async def summarize(self, llm, to_summarize: list,
                        existing_summary: Optional[str] = None) -> str:
        """LLM 生成滚动摘要：已有摘要并入压缩输入（滚动累积）

        摘要保留关键信息（预约槽位、偏好、结论），丢弃寒暄与冗余。
        """
        dialogue = "\n".join(f"{'用户' if m.type == 'human' else '助手'}：{m.content}"
                             for m in to_summarize)
        prompt = (f"请将以下对话历史压缩为简短摘要，保留关键信息"
                  f"（预约槽位、用户偏好、重要结论），丢弃寒暄与冗余表述，"
                  f"控制在100字以内。\n\n"
                  f"{'已有摘要：' + existing_summary if existing_summary else ''}\n"
                  f"待压缩对话：\n{dialogue}\n\n请直接输出摘要：")
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        summary = response.content if hasattr(response, "content") else str(response)
        return summary.strip()


def windowed_history_str(messages: list, summary: Optional[str] = None,
                         recent_count: Optional[int] = None) -> str:
    """构建窗口化历史文本：摘要前缀 + 最近 N 条消息（供提示词注入）"""
    from config import memory_config as mc
    recent_count = recent_count or mc.WINDOW_KEEP_AFTER_SUMMARY
    parts = []
    if summary:
        parts.append(f"【更早对话摘要】{summary}")
    for m in messages[-recent_count:]:
        role = "用户" if m.type == "human" else "助手"
        parts.append(f"{role}：{m.content}")
    return "\n".join(parts)


class SessionRegistry:
    """按 session_id 键控的会话注册表（带容量上限与 TTL 淘汰）"""

    MAX_SESSIONS = 1000          # 内存会话上限，超出按最近使用时间淘汰
    SESSION_TTL_SECONDS = 86400  # 24 小时未活跃的会话自动淘汰

    def __init__(self, db_path: str = None):
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

    def _evict_if_needed(self) -> None:
        """超容量淘汰最久未活跃会话；顺带清理过期会话（内存态；消息已落库可恢复）"""
        now = time.time()
        expired = [sid for sid, ctx in self._sessions.items()
                   if now - ctx.updated_at > self.SESSION_TTL_SECONDS]
        for sid in expired:
            self._sessions.pop(sid, None)
        if len(self._sessions) >= self.MAX_SESSIONS:
            oldest = sorted(self._sessions.items(), key=lambda kv: kv[1].updated_at)
            for sid, _ in oldest[:len(self._sessions) - self.MAX_SESSIONS + 1]:
                self._sessions.pop(sid, None)

    def get_or_create(self, session_id: Optional[str]) -> SessionContext:
        """获取或创建会话上下文；已存在则恢复历史消息"""
        if not session_id:
            session_id = self.new_session_id()
        ctx = self._sessions.get(session_id)
        if ctx is None:
            self._evict_if_needed()
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
