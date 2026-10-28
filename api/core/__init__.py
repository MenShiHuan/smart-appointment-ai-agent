"""
API核心组件初始化

导出核心的响应模型、异常处理等组件
"""
from .response_models import *
from .exceptions import *

__all__ = [
    # 响应模型
    "BaseResponse",
    "DataResponse",
    "AppointmentRequest",
    "AppointmentResponse",
    "ConsultationRequest",
    "ConsultationResponse",
    "UserBehaviorRequest",
    "UserBehaviorResponse",
    "TaskClassificationRequest",
    "TaskClassificationResponse",

    # 异常类
    "BusinessException",

    # 异常处理器
    "api_exception_handler",
    "http_exception_handler",
    "general_exception_handler",
]
