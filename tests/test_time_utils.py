"""
evals/time_utils.py 相对时间表达式解析的离线单测

基准日期固定注入，覆盖 TODAY / TOMORROW / TOMORROW+N / WEDNESDAY_NEXT / 裸星期 / 未知。
"""
from datetime import datetime

import pytest

from evals.time_utils import resolve_relative_time


# 固定基准：2026-09-24（周四）12:34
BASE = datetime(2026, 9, 24, 12, 34, 0)


@pytest.mark.parametrize("expr,expected", [
    ("TODAY 15:00", "2026-09-24 15:00"),
    ("TOMORROW 14:00", "2026-09-25 14:00"),
    ("TOMORROW+1 20:00", "2026-09-26 20:00"),
    ("TOMORROW+2 09:00", "2026-09-27 09:00"),
    ("WEDNESDAY_NEXT 10:00", "2026-09-30 10:00"),  # 下周三（周四基准）
    ("MONDAY_NEXT 09:00", "2026-09-28 09:00"),
    ("THURSDAY_NEXT 11:00", "2026-10-01 11:00"),   # 同名日 → 下一周
    ("FRIDAY 10:00", "2026-09-25 10:00"),          # 裸星期 = 最近的下一个周X
    ("THURSDAY 10:00", "2026-09-24 10:00"),        # 裸星期当天 → 当天
    ("TOMORROW", "2026-09-25 00:00"),
    ("未知", None),
    ("", None),
    ("2026-09-25 15:00", None),                     # 绝对时间不是相对表达式 → None
    ("NEXT_MONTH 10:00", None),                     # 不支持的表达式 → None
])
def test_resolve_relative_time(expr, expected):
    result = resolve_relative_time(expr, now=BASE)
    if expected is None:
        assert result is None
    else:
        assert result == datetime.strptime(expected, "%Y-%m-%d %H:%M")


def test_default_base_uses_beijing_time():
    """无注入基准时使用北京时间（与抽取提示词口径一致）"""
    from config.time_config import time_config
    result = resolve_relative_time("TODAY 00:00")
    today = time_config.now().replace(second=0, microsecond=0).replace(tzinfo=None)
    assert result == today.replace(hour=0, minute=0)
