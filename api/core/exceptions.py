"""
API 异常处理

统一错误响应格式：{"status": "error", "message": "..."}，
前端按 result.message 读取错误信息。
"""
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class BusinessException(HTTPException):
    """业务逻辑异常"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=message)


async def api_exception_handler(request: Request, exc: BusinessException):
    """业务异常处理器

    保留业务方设定的状态码（默认 400），按统一错误格式 {"status": "error", ...} 返回。
    """
    logger.error(f"业务异常: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 异常处理器：统一错误格式（含 422 校验错误等）"""
    detail = exc.detail
    if isinstance(detail, list):
        # Pydantic 校验错误列表 → 摘要字符串
        detail = "; ".join(str(d.get("msg", d)) for d in detail if isinstance(d, dict))
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": str(detail)}
    )


async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器

    500 兜底：对前端只回固定文案，异常详情与堆栈仅写入日志（避免内部信息泄漏）。
    """
    import traceback
    error_detail = f"未处理异常: {str(exc)}"
    stack_trace = traceback.format_exc()
    logger.error(f"{error_detail}\n{stack_trace}")

    # 固定文案兜底：不把异常细节透出给前端
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "服务器内部错误"}
    )
