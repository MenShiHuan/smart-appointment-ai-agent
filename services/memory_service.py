"""
离线记忆整合服务（C4a）

职责：
1. 触发判定（新增会话数 + 间隔时间，参数见 config/memory_config.py）
2. 增量回放行为事件 → 规则/LLM 混合抽取偏好候选
3. 向量去重、互斥类型冲突降权、置信度更新、固化写 UserPreference
4. 幂等保证：任务锁 + 水位检查点（批粒度推进）+ 变更日志

设计口径见 DEV_SPEC 2.6 / 2.6.1。
"""
import asyncio
import math
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import memory_config as mc
from config.model_provider import create_chat_model
from db.db_router import DatabaseRouter
from services.text_embedding import embed_input
from agents.user_behavior.memory_extractor import (
    ConsultationExtractor,
    PreferenceCandidate,
    extract_from_behavior,
)

DEFAULT_DB_PATH = "sqlite:///data/smart_appointment.db"


def _normalize(vector: List[float]) -> List[float]:
    """向量归一化（余弦相似度前置）"""
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class MemoryService:
    """离线记忆整合服务（按 user_id 维度；当前部署为单用户 default_user）"""

    def __init__(self, db_path: str = None, llm=None):
        self.db_router = DatabaseRouter(db_path)
        self.user_repo = self.db_router.user_behavior
        self.memory_repo = self.db_router.memory
        self.llm = llm or create_chat_model(temperature=0)
        self._lock = threading.Lock()

    # ============ 触发判定 ============

    @staticmethod
    def check_trigger(watermark: Optional[Dict[str, Any]], session_count: int,
                      now: datetime) -> tuple:
        """判定是否触发整合。返回 (should_run, reason)"""
        if watermark is None:
            if session_count < mc.MIN_SESSIONS:
                return False, f"会话数不足（当前 {session_count} < {mc.MIN_SESSIONS}）"
            return True, ""

        new_sessions = session_count - watermark.get("session_count_at_last_run", 0)
        if new_sessions < mc.MIN_SESSIONS:
            return False, f"新增会话不足（新增 {new_sessions} < {mc.MIN_SESSIONS}）"

        last_at = watermark.get("last_integration_at")
        if last_at is not None:
            elapsed = (now - last_at).total_seconds()
            if elapsed < mc.MIN_INTERVAL_HOURS * 3600:
                return False, f"间隔不足（距上次 {elapsed / 3600:.1f}h < {mc.MIN_INTERVAL_HOURS}h）"
        return True, ""

    # ============ 置信度 ============

    @staticmethod
    def compute_confidence(distinct_sessions: int, days_since_evidence: int) -> float:
        """置信度 = min(1, 基数 + 每会话增量×跨会话次数) × 日衰减系数^距上次证据天数"""
        base = min(1.0, mc.CONFIDENCE_BASE + mc.CONFIDENCE_PER_SESSION * distinct_sessions)
        decay = mc.CONFIDENCE_DECAY_PER_DAY ** max(days_since_evidence, 0)
        return base * decay

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """两个 embedding 的余弦相似度（调用前已归一化）"""
        return sum(x * y for x, y in zip(a, b))

    # ============ 主流程 ============

    async def run_integration_if_due(self, user_id: str = "default_user",
                                     now: Optional[datetime] = None) -> Dict[str, Any]:
        """触发判定 + 整合主流程（幂等，可重复调用）"""
        now = now or datetime.utcnow()
        session_count = self.user_repo.count_distinct_sessions(user_id)
        watermark = self.memory_repo.get_watermark(user_id)

        should_run, reason = self.check_trigger(watermark, session_count, now)
        if not should_run:
            return {"triggered": False, "reason": reason}

        if not self._lock.acquire(blocking=False):
            return {"triggered": False, "reason": "任务锁占用（已有整合在运行）"}

        try:
            return await self._run_integration(user_id, watermark, session_count, now)
        finally:
            self._lock.release()

    async def _run_integration(self, user_id: str, watermark: Optional[Dict[str, Any]],
                               session_count: int, now: datetime) -> Dict[str, Any]:
        stats = {"created": 0, "merged": 0, "downweighted": 0,
                 "retired": 0, "promoted": 0, "candidates": 0}
        processed = 0
        cursor = watermark["last_behavior_id"] if watermark else 0
        consultation_extractor = ConsultationExtractor(self.llm)

        while True:
            batch = self.user_repo.get_behaviors_after_id(user_id, cursor, mc.BATCH_SIZE)
            if not batch:
                break

            for behavior in batch:
                candidates = extract_from_behavior(behavior)
                data = behavior.get("action_data") or {}
                if behavior.get("action_type") == "consultation" and data.get("question"):
                    candidates += await consultation_extractor.extract(
                        data["question"], behavior)
                stats["candidates"] += len(candidates)
                for candidate in candidates:
                    self._apply_candidate(user_id, candidate, now, stats)
                processed += 1

            # 批粒度推进水位（先日志后水位已由各操作保证；此处提交即认为该批完成）
            cursor = batch[-1]["id"]
            self.memory_repo.save_watermark(user_id, cursor, session_count, now)

        return {
            "triggered": True,
            "reason": "",
            "processed_behaviors": processed,
            **stats,
        }

    # ============ 单候选应用：去重/合并/冲突降权/固化 ============

    def _apply_candidate(self, user_id: str, candidate: PreferenceCandidate,
                         now: datetime, stats: Dict[str, int]) -> None:
        embedding = _normalize(embed_input(candidate.content))
        active_entries = self.memory_repo.get_active_entries(user_id, candidate.entry_type)

        best_entry = None
        best_sim = -1.0
        for entry in active_entries:
            if not entry.get("embedding"):
                continue
            sim = self.cosine_similarity(embedding, _normalize(entry["embedding"]))
            if sim > best_sim:
                best_sim, best_entry = sim, entry

        if best_entry is not None and best_sim >= mc.EMBED_SIM_THRESHOLD:
            self._merge_candidate(user_id, best_entry, candidate, embedding, now, stats)
        else:
            self._create_candidate(user_id, candidate, embedding, now, stats)
            if candidate.entry_type in mc.MUTUALLY_EXCLUSIVE_TYPES:
                self._downweight_conflicts(user_id, active_entries, candidate, now, stats)

    def _merge_candidate(self, user_id: str, entry: Dict[str, Any],
                         candidate: PreferenceCandidate, embedding: List[float],
                         now: datetime, stats: Dict[str, int]) -> None:
        updated = self.memory_repo.merge_entry(entry["id"], candidate.session_id)
        distinct_sessions = updated["distinct_sessions"]
        confidence = self.compute_confidence(distinct_sessions, days_since_evidence=0)
        self.memory_repo.update_entry(entry["id"], confidence=confidence,
                                      embedding=embedding)
        self.memory_repo.add_change_log(user_id, "merge", entry["id"], {
            "content": candidate.content,
            "evidence_count_after": updated["evidence_count"],
            "distinct_sessions_after": distinct_sessions,
            "confidence_after": round(confidence, 4),
        })
        stats["merged"] += 1
        if confidence >= mc.PROMOTE_THRESHOLD:
            self._promote(user_id, entry, confidence, now, stats)

    def _create_candidate(self, user_id: str, candidate: PreferenceCandidate,
                          embedding: List[float], now: datetime,
                          stats: Dict[str, int]) -> None:
        confidence = self.compute_confidence(distinct_sessions=1, days_since_evidence=0)
        entry_id = self.memory_repo.create_entry(
            user_id, candidate.entry_type, candidate.content,
            embedding=embedding, confidence=confidence,
            session_id=candidate.session_id,
        )
        self.memory_repo.add_change_log(user_id, "create", entry_id, {
            "content": candidate.content,
            "entry_type": candidate.entry_type,
            "confidence": round(confidence, 4),
        })
        stats["created"] += 1
        if confidence >= mc.PROMOTE_THRESHOLD:
            entry = {"id": entry_id, "entry_type": candidate.entry_type,
                     "content": candidate.content, "status": "active"}
            self._promote(user_id, entry, confidence, now, stats)

    def _downweight_conflicts(self, user_id: str, active_entries: List[Dict[str, Any]],
                              candidate: PreferenceCandidate, now: datetime,
                              stats: Dict[str, int]) -> None:
        """互斥类型冲突：同类型既有条目（未被合并的）重要度减半，低于阈值退役"""
        for entry in active_entries:
            if not entry.get("embedding"):
                continue
            new_importance = (entry["importance"] or 1.0) * mc.IMPORTANCE_DOWNWEIGHT_FACTOR
            if new_importance < mc.RETIRE_IMPORTANCE_THRESHOLD:
                self.memory_repo.update_entry(entry["id"], importance=new_importance,
                                              status="retired")
                self.memory_repo.add_change_log(user_id, "retire", entry["id"], {
                    "content": entry["content"],
                    "new_evidence": candidate.content,
                    "importance_after": round(new_importance, 4),
                })
                stats["retired"] += 1
            else:
                self.memory_repo.update_entry(entry["id"], importance=new_importance)
                self.memory_repo.add_change_log(user_id, "downweight", entry["id"], {
                    "content": entry["content"],
                    "new_evidence": candidate.content,
                    "importance_after": round(new_importance, 4),
                })
                stats["downweighted"] += 1

    def _promote(self, user_id: str, entry: Dict[str, Any], confidence: float,
                 now: datetime, stats: Dict[str, int]) -> None:
        """固化：置信度达阈值 → 写 UserPreference（类型+值；置信度以 MemoryEntry 为单一事实源）"""
        self.user_repo.update_user_preference(
            user_id, entry["entry_type"], entry["content"])
        self.memory_repo.add_change_log(user_id, "promote", entry["id"], {
            "entry_type": entry["entry_type"],
            "content": entry["content"],
            "confidence": round(confidence, 4),
        })
        stats["promoted"] += 1

    # ============ C4b 加权召回 ============

    async def recall(self, user_id: str, query: str,
                     top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """分层记忆加权召回：得分 = 语义 0.6 + 时效 0.3 + 重要度 0.1，取 Top-K

        返回条目列表（含 score 与分项得分），按得分降序；无向量条目不参与语义分。
        """
        top_k = top_k or mc.RECALL_TOP_K
        entries = self.memory_repo.get_active_entries(user_id)
        if not entries:
            return []

        query_emb = _normalize(embed_input(query))
        now = datetime.utcnow()

        scored = []
        for entry in entries:
            semantic = 0.0
            if entry.get("embedding"):
                semantic = max(
                    0.0, self.cosine_similarity(query_emb, _normalize(entry["embedding"])))

            last_at = entry.get("last_evidence_at")
            days_since = max((now - last_at).days, 0) if last_at else 999
            recency = mc.CONFIDENCE_DECAY_PER_DAY ** days_since
            importance = min(entry.get("importance") or 0.0, 1.0)

            score = (mc.RECALL_WEIGHT_SEMANTIC * semantic
                     + mc.RECALL_WEIGHT_RECENCY * recency
                     + mc.RECALL_WEIGHT_IMPORTANCE * importance)
            scored.append({
                "entry_id": entry["id"],
                "entry_type": entry["entry_type"],
                "content": entry["content"],
                "confidence": entry["confidence"],
                "score": score,
                "semantic": round(semantic, 4),
                "recency": round(recency, 4),
                "importance": importance,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ============ 调度与手动触发入口 ============

    def run_integration_job(self, user_id: str = "default_user") -> Dict[str, Any]:
        """同步封装（供定时调度器使用）"""
        return asyncio.run(self.run_integration_if_due(user_id))


if __name__ == "__main__":
    # 手动触发入口：python -m services.memory_service
    service = MemoryService()
    report = service.run_integration_job()
    print("整合结果:", report)
