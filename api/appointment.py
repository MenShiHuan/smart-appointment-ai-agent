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
    """创建预约（走预约 Agent 完整链路：抽取 → 匹配 → 落库前确认）

    POST /api/appointment/create
    请求：AppointmentRequest，service_type/preferred_time/notes 拼接为一句自然语言交 Agent 抽取槽位
    响应：DataResponse，data 含 reply（Agent 流式输出的完整回复）与 session_id（REST 独立会话）
    错误：输入为空或 Agent 链路失败时返回 400，detail 透传失败原因
    """
    try:
        # 延迟导入全局单例：与其它 API 模块一致，首次请求时才加载 Agent
        from api.chat_handler import task_agent

        agent = task_agent.appointment_agent
        # 三个字段拼接为一句自然语言（空字段过滤掉），由 Agent 自行抽取槽位
        user_input = " ".join(filter(None, [
            request.service_type, request.preferred_time, request.notes
        ]))
        if not user_input:
            # 400：三字段全空，无有效输入，拒绝进入 Agent 链路
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
        # 已抛出的 HTTP 异常（如 400 空内容）原样透传，不再二次包装
        raise
    except Exception as e:
        # 其余异常统一转 400，异常详情透传给前端定位
        raise HTTPException(status_code=400, detail=str(e))
