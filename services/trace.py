"""
轻量可观测性（F1）：TraceContext + 指标聚合

- TraceContext：上下文管理器，记录阶段耗时与 Token 到 JSON Lines（logs/traces.jsonl）
- MetricsRegistry：进程内指标聚合（请求数/阶段耗时），供 /metrics 端点读取
"""
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

TRACE_FILE = Path("logs/traces.jsonl")
TRACE_LOCK = threading.Lock()


def _append_trace(record: dict) -> None:
    """追加一条 trace 记录（JSON Lines；文件写入失败不阻断业务）"""
    try:
        TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_LOCK:
            with open(TRACE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


class MetricsRegistry:
    """进程内指标聚合：请求计数与阶段耗时统计"""

    def __init__(self):
        self._lock = threading.Lock()
        self.request_count = 0
        self.error_count = 0
        self._stage_elapsed: Dict[str, List[float]] = {}

    def record(self, stage: str, elapsed: float, is_error: bool = False) -> None:
        """记录一次阶段执行：请求计数 +1，错误计入错误计数，耗时按阶段聚合"""
        with self._lock:
            self.request_count += 1
            if is_error:
                self.error_count += 1
            self._stage_elapsed.setdefault(stage, []).append(elapsed)

    def snapshot(self) -> dict:
        """返回指标快照：请求/错误计数与各阶段耗时统计（次数、总耗时、均值、最大值）"""
        with self._lock:
            stages = {}
            for stage, values in self._stage_elapsed.items():
                if not values:
                    continue
                stages[stage] = {
                    "count": len(values),
                    "total_seconds": round(sum(values), 3),
                    "avg_seconds": round(sum(values) / len(values), 3),
                    "max_seconds": round(max(values), 3),
                }
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "stages": stages,
            }


# 进程级指标注册表（/metrics 端点读取）
metrics_registry = MetricsRegistry()


class TraceContext:
    """阶段级 Trace：记录阶段名、耗时、Token 数；可选关联 session_id"""

    def __init__(self, stage: str, session_id: Optional[str] = None):
        self.stage = stage
        self.session_id = session_id
        self.trace_id = uuid.uuid4().hex[:16]
        self._start = None
        self.token_count = 0

    def __enter__(self) -> "TraceContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.perf_counter() - self._start
        is_error = exc_type is not None
        metrics_registry.record(self.stage, elapsed, is_error=is_error)
        _append_trace({
            "trace_id": self.trace_id,
            "stage": self.stage,
            "session_id": self.session_id,
            "elapsed_seconds": round(elapsed, 4),
            "token_count": self.token_count,
            "error": str(exc) if is_error else None,
        })

    def count_tokens(self, n: int = 1) -> None:
        """累计输出 token 数（流式路径逐块调用）"""
        self.token_count += n


@contextmanager
def trace_stage(stage: str, session_id: Optional[str] = None):
    """便捷上下文管理器：with trace_stage("consultation"): ..."""
    with TraceContext(stage, session_id) as ctx:
        yield ctx
