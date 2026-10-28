"""
简化的任务分类API

只保留第一版核心功能
"""
from fastapi import APIRouter, HTTPException
from .core.response_models import (
    TaskClassificationRequest,
    TaskClassificationResponse,
    DataResponse
)

router = APIRouter(prefix="/api/task", tags=["任务分类"])


@router.post("/classify", response_model=DataResponse)
async def classify_task(request: TaskClassificationRequest):
    """分类任务

    POST /api/task/classify
    请求：TaskClassificationRequest（text 待分类文本）
    响应：DataResponse，data 含分类标签（appointment/query/memory/other 之一）；失败时返回 400。
    """
    try:
        # 复用全局单例主控 Agent（会话状态按 session_id 隔离）
        from api.chat_handler import task_agent
        category = await task_agent.task_classifier.classify_task(request.text)

        return DataResponse(
            message="任务分类成功",
            data={"text": request.text, "category": category}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
