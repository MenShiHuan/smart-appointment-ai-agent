"""
数据库会话管理

SQLite 引擎创建、事务作用域管理与进程级共享单例（按 db_path 键控）。
"""
import os
import threading
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from ..models import Base

# 默认 SQLite 库文件（未配置 DATABASE_URL 时使用）
DEFAULT_DB_PATH = 'sqlite:///data/smart_appointment.db'

# 进程级 SessionManager 共享缓存：同一 db_path 只建一个 engine（含一次 create_all），
# 避免多 Service 各自建 engine 造成写锁竞争与重复 DDL；按路径键控支持评测独立库。
_shared_managers = {}
_shared_lock = threading.Lock()


def resolve_db_path(db_path=None) -> str:
    """解析数据库路径：显式参数 > DATABASE_URL 环境变量 > 默认路径"""
    return db_path or os.environ.get("DATABASE_URL") or DEFAULT_DB_PATH


def get_session_manager(db_path=None) -> "SessionManager":
    """获取进程级共享的会话管理器（懒加载单例，按 db_path 键控）"""
    path = resolve_db_path(db_path)
    with _shared_lock:
        manager = _shared_managers.get(path)
        if manager is None:
            manager = SessionManager(path)
            _shared_managers[path] = manager
        return manager


class SessionManager:
    """
    数据库会话管理器

    职责：
    1. 管理数据库连接和会话
    2. 提供统一的会话上下文管理
    3. 处理事务和异常回滚

    优先使用 get_session_manager() 获取进程级共享实例。
    """

    def __init__(self, db_path=None):
        """
        初始化会话管理器

        Args:
            db_path: 数据库连接路径（None 时取 DATABASE_URL 环境变量或默认路径）
        """
        self.db_path = resolve_db_path(db_path)
        # timeout 缓解 SQLite 写锁竞争（"database is locked"）
        self.engine = create_engine(self.db_path, connect_args={"timeout": 30})
        # 建表（create_all 幂等：已存在的表不会重复创建）
        Base.metadata.create_all(self.engine)
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    @contextmanager
    def session_scope(self):
        """
        提供会话上下文管理

        自动处理：
        - 会话创建和关闭
        - 事务提交和回滚
        - 异常处理
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self):
        """关闭会话管理器"""
        self.Session.remove()
        self.engine.dispose()
