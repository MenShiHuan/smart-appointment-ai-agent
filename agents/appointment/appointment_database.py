"""
预约数据库操作器

负责处理预约相关的数据库操作
注意：现在通过Services层访问数据库，符合分层架构
"""

from typing import Dict, Any
from datetime import datetime
from config.time_config import time_config


class AppointmentDatabase:
    """预约数据库操作器"""
    
    def __init__(self):
        # 延迟导入Services避免循环依赖
        self._appointment_service = None
        self._user_behavior_service = None
    
    @property
    def appointment_service(self):
        """懒加载预约服务"""
        if self._appointment_service is None:
            from services.appointment_service import AppointmentService
            self._appointment_service = AppointmentService()
        return self._appointment_service
    
    @property 
    def user_behavior_service(self):
        """懒加载用户行为服务"""
        if self._user_behavior_service is None:
            from services.user_behavior_service import UserBehaviorService
            self._user_behavior_service = UserBehaviorService()
        return self._user_behavior_service
    
    def save_appointment(self, technician_id: str, start_time: datetime, 
                        end_time: datetime, appointment_history: Dict[str, Any], 
                        session_id: str) -> bool:
        """保存预约信息到数据库

        成功后同步记录用户行为（埋点）；任何异常都兜底返回 False 而不上抛，
        保证持久化失败不中断预约流程。
        """
        try:
            # 通过Services层保存预约
            success = self.appointment_service.save_appointment(
                technician_id, start_time, end_time, appointment_history, session_id
            )
            
            if success:
                # 记录用户行为
                self._record_user_behavior(start_time, end_time, technician_id, 
                                         appointment_history, session_id)
            
            return success
            
        except Exception as e:
            print(f"保存预约信息到数据库失败：{e}")
            return False
    
    def _record_user_behavior(self, start_time: datetime, end_time: datetime,
                            technician_id: str, appointment_history: Dict[str, Any],
                            session_id: str):
        """记录用户预约行为

        记录失败仅打印日志、不影响预约成功结果（埋点属于旁路逻辑）。
        """
        try:
            action_data = {
                'start_time': time_config.format_datetime(start_time, "%Y-%m-%d %H:%M:%S"),
                'end_time': time_config.format_datetime(end_time, "%Y-%m-%d %H:%M:%S"),
                'duration': int((end_time - start_time).total_seconds() / 60),
                'project': appointment_history.get('project', '未知'),
                'preference': appointment_history.get('preference', ''),
                'technician_id': technician_id
            }

            # 通过Services层记录用户行为（technician_id 列类型为 Integer，传 int）
            self.user_behavior_service.record_behavior(
                user_id="default_user",  # 统一使用default_user作为用户ID
                action_type='appointment',
                action_data=action_data,
                technician_id=int(technician_id),
                session_id=session_id
            )
            
        except Exception as behavior_error:
            print(f"记录用户行为失败（但预约仍然成功）：{behavior_error}")
