"""
记忆条目数据访问对象（C4a 离线记忆整合）

职责：
1. MemoryEntry 的创建、合并（证据累加）、降权与退役
2. IntegrationWatermark 水位检查点的读写
3. MemoryChangeLog 变更日志的追加（幂等审计）

数据层设计口径见 DEV_SPEC 2.6.1。
"""
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..base.session_manager import SessionManager
from ..models import utcnow, MemoryEntry, IntegrationWatermark, MemoryChangeLog


def _entry_to_dict(entry: MemoryEntry) -> Dict[str, Any]:
    return {
        "id": entry.id,
        "user_id": entry.user_id,
        "entry_type": entry.entry_type,
        "content": entry.content,
        "embedding": entry.embedding,
        "importance": entry.importance,
        "confidence": entry.confidence,
        "evidence_count": entry.evidence_count,
        "distinct_sessions": entry.distinct_sessions,
        "last_evidence_at": entry.last_evidence_at,
        "status": entry.status,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
    }


class MemoryRepository:
    """记忆条目 / 水位 / 变更日志的数据访问层"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    # ============ MemoryEntry ============

    def create_entry(self, user_id: str, entry_type: str, content: str,
                     embedding: Optional[List[float]] = None, importance: float = 1.0,
                     confidence: float = 0.0, session_id: Optional[str] = None) -> int:
        """创建记忆条目，返回条目 id（初始证据来自 session_id 对应会话）"""
        sessions = [session_id] if session_id else []
        with self.session_manager.session_scope() as session:
            entry = MemoryEntry(
                user_id=user_id,
                entry_type=entry_type,
                content=content,
                embedding=embedding,
                importance=importance,
                confidence=confidence,
                distinct_sessions=len(sessions),
                sessions=sessions,
            )
            session.add(entry)
            session.flush()
            return entry.id

    def get_active_entries(self, user_id: str, entry_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取用户 active 状态的记忆条目（召回与去重共用）"""
        with self.session_manager.session_scope() as session:
            query = session.query(MemoryEntry).filter(
                MemoryEntry.user_id == user_id,
                MemoryEntry.status == "active",
            )
            if entry_type:
                query = query.filter(MemoryEntry.entry_type == entry_type)
            return [_entry_to_dict(e) for e in query.all()]

    def merge_entry(self, entry_id: int, session_id: Optional[str],
                    importance: Optional[float] = None,
                    confidence: Optional[float] = None,
                    now: Optional[datetime] = None) -> Dict[str, Any]:
        """合并证据：证据次数 +1，追加新会话、刷新证据时间与统计值"""
        now = now or utcnow()
        with self.session_manager.session_scope() as session:
            entry = session.get(MemoryEntry, entry_id)
            if entry is None:
                raise ValueError(f"记忆条目不存在: {entry_id}")
            entry.evidence_count += 1
            sessions = list(entry.sessions or [])
            if session_id and session_id not in sessions:
                sessions.append(session_id)
            entry.sessions = sessions
            entry.distinct_sessions = len(sessions)
            entry.last_evidence_at = now
            if importance is not None:
                entry.importance = importance
            if confidence is not None:
                entry.confidence = confidence
            session.flush()
            return _entry_to_dict(entry)

    def update_entry(self, entry_id: int, importance: Optional[float] = None,
                     confidence: Optional[float] = None,
                     status: Optional[str] = None,
                     embedding: Optional[List[float]] = None) -> None:
        """更新条目统计字段（降权/退役/置信度刷新/向量刷新）；条目不存在抛 ValueError"""
        with self.session_manager.session_scope() as session:
            entry = session.get(MemoryEntry, entry_id)
            if entry is None:
                raise ValueError(f"记忆条目不存在: {entry_id}")
            if importance is not None:
                entry.importance = importance
            if confidence is not None:
                entry.confidence = confidence
            if status is not None:
                entry.status = status
            if embedding is not None:
                entry.embedding = embedding

    # ============ IntegrationWatermark ============

    def get_watermark(self, user_id: str) -> Optional[Dict[str, Any]]:
        """读取水位检查点；无记录返回 None"""
        with self.session_manager.session_scope() as session:
            wm = session.get(IntegrationWatermark, user_id)
            if wm is None:
                return None
            return {
                "user_id": wm.user_id,
                "last_behavior_id": wm.last_behavior_id,
                "last_integration_at": wm.last_integration_at,
                "session_count_at_last_run": wm.session_count_at_last_run,
            }

    def save_watermark(self, user_id: str, last_behavior_id: int,
                       session_count: int, now: Optional[datetime] = None) -> None:
        """写入（upsert）水位检查点"""
        now = now or utcnow()
        with self.session_manager.session_scope() as session:
            wm = session.get(IntegrationWatermark, user_id)
            if wm is None:
                wm = IntegrationWatermark(user_id=user_id)
                session.add(wm)
            wm.last_behavior_id = last_behavior_id
            wm.last_integration_at = now
            wm.session_count_at_last_run = session_count

    # ============ MemoryChangeLog ============

    def add_change_log(self, user_id: str, operation: str, entry_id: int,
                       detail: Optional[Dict[str, Any]] = None) -> int:
        """追加变更日志（幂等审计），返回日志 id"""
        with self.session_manager.session_scope() as session:
            log = MemoryChangeLog(
                user_id=user_id,
                operation=operation,
                entry_id=entry_id,
                detail=detail,
            )
            session.add(log)
            session.flush()
            return log.id
