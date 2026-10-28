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
    """提交咨询问题

    POST /api/consultation/ask
    一次性非流式咨询（无会话历史），LLM 结合知识库检索生成回答；
    响应：DataResponse，data 含 answer 与原始 question；失败时返回 400。
    """
    try:
        # 延迟导入全局单例咨询 Agent（实例本身无状态，REST 场景无需会话）
        from api.chat_handler import task_agent

        agent = task_agent.consultant_agent
        result = await agent.consult(request.question)

        return DataResponse(
            message="咨询处理成功",
            data={"answer": result, "question": request.question}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
