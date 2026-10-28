"""
会话消息服务层

职责：
1. 封装会话消息的数据库操作
2. 供 Agents 层（SessionRegistry）访问，遵守五层架构（Agents → Services → DB）
"""
from typing import List, Dict, Any, Optional
from db.db_router import DatabaseRouter


class ConversationService:
    """会话消息服务"""

    def __init__(self, db_path: str = None):
        self.db_router = DatabaseRouter(db_path)

    @property
    def conversation_repo(self):
        return self.db_router.conversation

    def add_message(self, session_id: str, role: str, content: str) -> int:
        """添加一条会话消息"""
        # 返回新消息的数据库 ID
        return self.conversation_repo.add_message(session_id, role, content)

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """按时间顺序获取会话消息"""
        return self.conversation_repo.get_messages(session_id, limit)

    def delete_session_messages(self, session_id: str) -> int:
        """删除会话的全部消息"""
        return self.conversation_repo.delete_session_messages(session_id)
