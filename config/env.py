"""
环境变量读取统一助手（config 层内部工具）

memory_config / vector_config 等配置模块共用；语义：
值为空串或未设置时返回默认值。
"""
import os


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default
