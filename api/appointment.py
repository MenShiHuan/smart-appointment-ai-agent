"""
简化的预约API

只保留第一版核心功能
"""
from fastapi import APIRouter, HTTPException
from .core.response_models import (
    AppointmentRequest,
    AppointmentResponse,
    DataResponse
)

router = APIRouter(prefix="/api/appointment", tags=["预约管理"])


@router.post("/create", response_model=DataResponse)
async def create_appointment(request: AppointmentRequest):
    """创建预约（走预约 Agent 完整链路：抽取 → 匹配 → 落库前确认）"""
    try:
        from api.chat_handler import task_agent

        agent = task_agent.appointment_agent
        user_input = " ".join(filter(None, [
            request.service_type, request.preferred_time, request.notes
        ]))
        if not user_input:
            raise HTTPException(status_code=400, detail="预约内容不能为空")

        # 独立会话（REST 请求无多轮上下文）
        session = task_agent.session_registry.get_or_create(None)
        tokens = []
        async for token in agent.run_stream(user_input=user_input, session=session):
            tokens.append(token)

        return DataResponse(
            message="预约处理完成",
            data={"reply": "".join(tokens), "session_id": session.session_id}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
