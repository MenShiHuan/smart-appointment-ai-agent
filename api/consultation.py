"""
简化的咨询API

只保留第一版核心功能
"""
from fastapi import APIRouter, HTTPException
from .core.response_models import (
    ConsultationRequest,
    ConsultationResponse,
    DataResponse
)

router = APIRouter(prefix="/api/consultation", tags=["咨询服务"])


@router.post("/ask", response_model=DataResponse)
async def ask_consultation(request: ConsultationRequest):
    """提交咨询问题"""
    try:
        from api.chat_handler import task_agent

        agent = task_agent.consultant_agent
        result = await agent.consult(request.question)

        return DataResponse(
            message="咨询处理成功",
            data={"answer": result, "question": request.question}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
