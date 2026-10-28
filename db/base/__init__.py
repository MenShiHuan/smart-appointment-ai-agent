"""
Database Base Module

数据库基础模块，包含：
- 会话管理
- 抽象接口
- 通用工具
"""

from .session_manager import (
    SessionManager,
    get_session_manager,
    resolve_db_path,
    DEFAULT_DB_PATH,
)
from .interfaces import (BaseTechnicianRepository, BaseKnowledgeRepository,
                         BaseUserBehaviorRepository, BaseScheduleRepository)

__all__ = [
    'SessionManager',
    'get_session_manager',
    'resolve_db_path',
    'DEFAULT_DB_PATH',
    'BaseTechnicianRepository',
    'BaseKnowledgeRepository',
    'BaseUserBehaviorRepository',
    'BaseScheduleRepository'
]
