"""
简化的API响应模型

只保留第一版真正需要的核心功能
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from datetime import datetime
from config.time_config import time_config


class BaseResponse(BaseModel):
    """基础响应模型

    timestamp 统一取 time_config.now()（北京时间口径，与全站时间基准一致）。
    """
    message: str
    timestamp: datetime = Field(default_factory=time_config.now)


class DataResponse(BaseResponse):
    """数据响应模型

    通用成功响应包装：data 承载任意业务负载，message 为提示文案。
    """
    data: Any


# 预约相关模型
class AppointmentRequest(BaseModel):
    """预约创建请求：service_type 服务类型、preferred_time 期望时间、notes 备注（可选）"""
    user_id: str
    service_type: str
    preferred_time: str
    notes: Optional[str] = None


class AppointmentResponse(BaseModel):
    """预约结果响应：appointment_id 与 scheduled_time/status 等落库结果"""
    appointment_id: str
    user_id: str
    service_type: str
    scheduled_time: str
    status: str
    notes: Optional[str] = None


# 咨询相关模型
class ConsultationRequest(BaseModel):
    """咨询请求：question 必填，category 为可选分类提示"""
    user_id: str
    question: str
    category: Optional[str] = None


class ConsultationResponse(BaseModel):
    """咨询响应：answer 为咨询回答，category 为命中的分类"""
    consultation_id: str
    question: str
    answer: str
    category: Optional[str] = None


# 用户行为相关模型
class UserBehaviorRequest(BaseModel):
    """用户行为上报请求：action 行为类型，context 附加上下文"""
    user_id: str
    action: str
    context: Optional[Dict[str, Any]] = None


class UserBehaviorResponse(BaseModel):
    """用户行为上报响应：回显行为与时间戳"""
    user_id: str
    action: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None


# 任务分类相关模型
class TaskClassificationRequest(BaseModel):
    """任务分类请求：text 待分类文本"""
    text: str
    context: Optional[Dict[str, Any]] = None


class TaskClassificationResponse(BaseModel):
    """任务分类响应：category 分类结果、confidence 置信度、reasoning 分类理由（可选）"""
    text: str
    category: str
    confidence: float
    reasoning: Optional[str] = None
