"""
预约信息抽取的结构化输出模型

用于 InputParser 的结构化输出路径（with_structured_output / JSON 校验）。
字段与提示词模板中的 JSON 约定一一对应，默认值为"未知"以兼容降级逻辑。
"""
from typing import List
from pydantic import BaseModel, Field


class AppointmentExtraction(BaseModel):
    """预约信息抽取结果"""

    gender: str = Field(default="未知", description="技师/理疗师性别偏好（男/女/未知）")
    start_time: str = Field(default="未知", description="预约起始时间，标准格式 YYYY-MM-DD HH:MM")
    duration: str = Field(default="未知", description="服务时长，统一为分钟数格式（如 60分钟）")
    project: str = Field(default="未知", description="服务项目（如肩颈理疗/艾灸）")
    preference: str = Field(default="未知", description="用户倾向（如手法轻重偏好）")
    technician_name: str = Field(default="未知", description="指定的技师/理疗师姓名")
    confirmation: str = Field(default="未知", description="用户对推荐确认问题的回复内容（是/好/可以/不/不要等）")
    info_complete: bool = Field(default=False, description="必需信息是否齐全")
    unrelated: bool = Field(default=False, description="用户问题是否与预约无关")
    missing_info: List[str] = Field(default_factory=list, description="缺失的关键信息字段列表")
