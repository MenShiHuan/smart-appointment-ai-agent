"""
评测集相对时间表达式解析（E2）

数据集中的时间期望值使用相对表达式，评测脚本按运行日换算，
保证数据集永不过期。支持：
- TODAY 15:00           今天
- TOMORROW 14:00        明天
- TOMORROW+N 20:00      后天及以后（N 天后）
- WEDNESDAY_NEXT 10:00  最近的下一个周X
"""
import re
from datetime import datetime, timedelta
from typing import Optional

WEEKDAYS = {"MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
            "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6}

_PATTERN = re.compile(
    r"^(TODAY|TOMORROW(?:\+\d+)?|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY|"
    r"MONDAY_NEXT|TUESDAY_NEXT|WEDNESDAY_NEXT|THURSDAY_NEXT|FRIDAY_NEXT|SATURDAY_NEXT|SUNDAY_NEXT)"
    r"(?:\s+(\d{1,2}):(\d{2}))?$")


def resolve_relative_time(expr: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """解析相对时间表达式为绝对 datetime（北京墙钟 naive）；无法解析返回 None

    基准时间默认取 time_config 的北京时间——与抽取提示词注入的"当前北京时间"一致，
    避免在非 UTC+8 机器上期望换算与模型所见日期错位。
    """
    if not expr or expr in ("未知", ""):
        return None
    match = _PATTERN.match(expr.strip())
    if not match:
        return None

    if now is None:
        from config.time_config import time_config
        now = time_config.now()
    base = now.replace(second=0, microsecond=0).replace(tzinfo=None)
    token = match.group(1)

    if token == "TODAY":
        day = base
    elif token == "TOMORROW":
        day = base + timedelta(days=1)
    elif token.startswith("TOMORROW+"):
        # TOMORROW+N = 明天之后再推 N 天（TOMORROW+1 即后天）
        offset = int(token.split("+")[1])
        day = base + timedelta(days=offset + 1)
    else:
        target_weekday = WEEKDAYS[token.replace("_NEXT", "")]
        days_ahead = (target_weekday - base.weekday()) % 7
        if token.endswith("_NEXT") and days_ahead == 0:
            days_ahead = 7  # "_NEXT" 语义：至少是下一周的同名日
        day = base + timedelta(days=days_ahead)

    hour = int(match.group(2)) if match.group(2) else 0
    minute = int(match.group(3)) if match.group(3) else 0
    return day.replace(hour=hour, minute=minute)
