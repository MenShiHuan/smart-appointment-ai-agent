"""
用户行为分析API
"""
import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user-behavior", tags=["用户行为分析"])


class UserAnalysisResponse(BaseModel):
    """用户分析响应"""
    favorite_technician_id: Optional[int] = None
    favorite_technician_name: Optional[str] = None
    favorite_service: Optional[str] = None
    favorite_duration: Optional[int] = None
    total_appointments: int = 0
    days_since_last_appointment: Optional[int] = None
    should_send_reminder: bool = False


async def get_user_analysis(user_id: str = "default_user") -> UserAnalysisResponse:
    """获取用户行为分析数据"""
    try:
        # 复用全局单例（避免每请求重建 LLM 客户端与数据服务）
        from api.chat_handler import task_agent
        agent = task_agent.behavior_agent
        analysis = agent.get_user_analysis(user_id)

        if not analysis:
            return UserAnalysisResponse()

        # 获取理疗师姓名（经 Services 层，遵守五层架构）
        technician_name = None
        if analysis.get('favorite_technician_id'):
            tech_info = agent.technician_service.get_technician_by_id(
                analysis['favorite_technician_id'])
            if tech_info:
                technician_name = tech_info.get('name')

        return UserAnalysisResponse(
            favorite_technician_id=analysis.get('favorite_technician_id'),
            favorite_technician_name=technician_name,
            favorite_service=analysis.get('favorite_service'),
            favorite_duration=analysis.get('favorite_duration'),
            total_appointments=analysis.get('total_appointments', 0),
            days_since_last_appointment=analysis.get('days_since_last_appointment'),
            should_send_reminder=analysis.get('should_send_reminder', False)
        )
    except Exception as e:
        # 记录异常日志并返回空数据（监控可见），而不是硬编码假数据
        logger.error(f"获取用户分析数据失败: {e}")
        return UserAnalysisResponse()


@router.get("/analysis", response_model=UserAnalysisResponse)
async def get_default_user_analysis():
    """获取默认用户的行为分析数据"""
    return await get_user_analysis("default_user")


class ReminderRequest(BaseModel):
    """发送提醒请求"""
    user_id: str = "default_user"


class ReminderResponse(BaseModel):
    """提醒消息响应"""
    message: str
    technician_available_times: Optional[List[dict]] = None


@router.post("/send-reminder", response_model=ReminderResponse, summary="发送回访提醒")
async def send_reminder(request: ReminderRequest):
    """生成并返回回访提醒消息"""
    try:
        from api.chat_handler import task_agent
        agent = task_agent.behavior_agent
        result = await agent.get_reminder_with_schedule(request.user_id)

        return ReminderResponse(
            message=result["message"],
            technician_available_times=result["technician_available_times"]
        )

    except Exception as e:
        logger.error(f"生成回访提醒失败: {e}")
        return ReminderResponse(
            message="尊敬的客户，您好！系统暂时无法查询理疗师时间，请稍后再试或直接联系我们预约。",
            technician_available_times=[]
        )
