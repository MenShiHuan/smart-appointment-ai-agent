"""
会话消息数据访问对象

职责：
1. 会话消息的持久化（写入/读取/删除）
2. 支持按 session_id 隔离
3. 支撑会话恢复（服务重启后重建聊天历史）
"""
from typing import List, Dict, Any, Optional
from ..base.session_manager import SessionManager
from ..models import ConversationMessage


class ConversationRepository:
    """会话消息仓库"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def add_message(self, session_id: str, role: str, content: str) -> int:
        """添加一条会话消息"""
        with self.session_manager.session_scope() as session:
            message = ConversationMessage(
                session_id=session_id,
                role=role,
                content=content
            )
            session.add(message)
            # flush 让自增 ID 立即可用，供调用方直接使用
            session.flush()
            return message.id

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """按时间顺序获取会话消息"""
        with self.session_manager.session_scope() as session:
            query = (
                session.query(ConversationMessage)
                .filter(ConversationMessage.session_id == session_id)
                # 按写入时间升序；同刻消息以 id 兜底，保证回放顺序稳定
                .order_by(ConversationMessage.created_at, ConversationMessage.id)
            )
            if limit is not None:
                query = query.limit(limit)
            return [
                {
                    "session_id": m.session_id,
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at,
                }
                for m in query.all()
            ]

    def delete_session_messages(self, session_id: str) -> int:
        """删除会话的全部消息（用于清空对话）"""
        with self.session_manager.session_scope() as session:
            count = (
                session.query(ConversationMessage)
                .filter(ConversationMessage.session_id == session_id)
                .delete(synchronize_session=False)
            )
            return count
