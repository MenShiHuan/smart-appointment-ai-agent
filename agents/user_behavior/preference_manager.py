"""
偏好管理器 - 专门负责管理用户的偏好数据

职责：
1. 从预约数据中提取和更新用户偏好
2. 管理技师偏好、时间偏好、服务偏好等
3. 提供偏好数据的查询和统计
4. 处理偏好的变化和趋势分析
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

# 偏好管理器内部统一使用的用户 ID（与 BehaviorRecorder/PatternAnalyzer 保持一致，
# 真实多用户体系的用户 ID 打通属后续改造，当前阶段约定单一默认用户）
DEFAULT_USER_ID = "default_user"


class PreferenceManager:
    """偏好管理器 - 负责用户偏好的管理和分析"""

    def __init__(self, behavior_service = None):
        """
        初始化偏好管理器

        Args:
            behavior_service: 用户行为服务实例
        """
        self.behavior_service = behavior_service
        self.logger = logging.getLogger(__name__)

    @property
    def behavior_db(self):
        """为了向后兼容，提供behavior_db属性（转发到 Service 层，遵守五层架构）"""
        if hasattr(self, 'behavior_service') and self.behavior_service:
            # 返回一个适配器，将调用转发到behavior_service
            return self.behavior_service
        else:
            # 如果没有service，返回None（应该在组件初始化时提供适当的处理）
            return None
    
    def update_technician_preference(self, technician_id: int):
        """
        更新技师偏好
        
        Args:
            technician_id: 技师ID
        """
        try:
            self.behavior_db.update_user_preference(DEFAULT_USER_ID, 'technician', str(technician_id))
            self.logger.info(f"更新技师偏好: {technician_id}")
        except Exception as e:
            self.logger.error(f"更新技师偏好失败: {str(e)}")
    
    def update_time_preference(self, start_time: str):
        """
        更新时间偏好
        
        Args:
            start_time: 开始时间字符串
        """
        try:
            start_datetime = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            hour = start_datetime.hour
            
            # 时段边界与 MemoryExtractor 的 _time_period_from_start_time 保持一致
            if 6 <= hour < 12:
                time_period = '上午'
            elif 12 <= hour < 18:
                time_period = '下午'
            else:
                time_period = '晚上'
            
            self.behavior_db.update_user_preference(DEFAULT_USER_ID, 'time_period', time_period)
            self.logger.info(f"更新时间偏好: {time_period}")
        except Exception as e:
            self.logger.error(f"更新时间偏好失败: {str(e)}")
    
    def update_duration_preference(self, duration: Any):
        """
        更新服务时长偏好
        
        Args:
            duration: 服务时长
        """
        try:
            duration_str = str(duration)
            self.behavior_db.update_user_preference(DEFAULT_USER_ID, 'duration', duration_str)
            self.logger.info(f"更新时长偏好: {duration_str}")
        except Exception as e:
            self.logger.error(f"更新时长偏好失败: {str(e)}")
    
    def update_service_preference(self, service: str):
        """
        更新服务项目偏好
        
        Args:
            service: 服务项目
        """
        try:
            self.behavior_db.update_user_preference(DEFAULT_USER_ID, 'service', service)
            self.logger.info(f"更新服务偏好: {service}")
        except Exception as e:
            self.logger.error(f"更新服务偏好失败: {str(e)}")
    
    def update_technician_type_preference(self, technician_type: str):
        """
        更新技师类型偏好
        
        Args:
            technician_type: 技师类型偏好（如：力气大、手法轻等）
        """
        try:
            self.behavior_db.update_user_preference(DEFAULT_USER_ID, 'technician_type', technician_type)
            self.logger.info(f"更新技师类型偏好: {technician_type}")
        except Exception as e:
            self.logger.error(f"更新技师类型偏好失败: {str(e)}")
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """
        获取用户所有偏好
        
        Returns:
            Dict: 用户偏好数据
        """
        try:
            # Service 层返回偏好行列表（按置信度降序），转换为 {类型: 值} 字典
            rows = self.behavior_db.get_user_preferences(DEFAULT_USER_ID)
            return {
                row['preference_type']: row['preference_value']
                for row in rows
                if row.get('preference_value')
            }
        except Exception as e:
            self.logger.error(f"获取用户偏好失败: {str(e)}")
            return {}
    
    def get_preferred_technician_id(self) -> Optional[int]:
        """
        获取偏好的技师ID
        
        Returns:
            int: 技师ID，如果没有偏好则返回None
        """
        try:
            preferences = self.get_user_preferences()
            technician_id = preferences.get('technician')
            return int(technician_id) if technician_id else None
        except Exception as e:
            self.logger.error(f"获取偏好技师ID失败: {str(e)}")
            return None
    
    def get_preferred_time_period(self) -> Optional[str]:
        """
        获取偏好的时间段
        
        Returns:
            str: 时间段偏好
        """
        try:
            preferences = self.get_user_preferences()
            return preferences.get('time_period')
        except Exception as e:
            self.logger.error(f"获取偏好时间段失败: {str(e)}")
            return None
    
    def get_preferred_service(self) -> Optional[str]:
        """
        获取偏好的服务项目
        
        Returns:
            str: 服务项目偏好
        """
        try:
            preferences = self.get_user_preferences()
            return preferences.get('service')
        except Exception as e:
            self.logger.error(f"获取偏好服务失败: {str(e)}")
            return None
    
    def get_preference_summary(self) -> Dict[str, Any]:
        """
        获取偏好摘要信息
        
        Returns:
            Dict: 偏好摘要
        """
        try:
            preferences = self.get_user_preferences()
            
            summary = {
                'has_technician_preference': bool(preferences.get('technician')),
                'has_time_preference': bool(preferences.get('time_period')),
                'has_service_preference': bool(preferences.get('service')),
                'has_duration_preference': bool(preferences.get('duration')),
                'preference_count': len([v for v in preferences.values() if v])
            }
            
            # 添加具体偏好内容
            if preferences.get('technician'):
                summary['preferred_technician_id'] = int(preferences['technician'])
            if preferences.get('time_period'):
                summary['preferred_time'] = preferences['time_period']
            if preferences.get('service'):
                summary['preferred_service'] = preferences['service']
            if preferences.get('duration'):
                summary['preferred_duration'] = preferences['duration']
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取偏好摘要失败: {str(e)}")
            return {}
    
    def clear_preference(self, preference_type: str):
        """
        清除特定类型的偏好
        
        Args:
            preference_type: 偏好类型
        """
        try:
            self.behavior_db.delete_user_preference(DEFAULT_USER_ID, preference_type)
            self.logger.info(f"清除偏好: {preference_type}")
        except Exception as e:
            self.logger.error(f"清除偏好失败: {str(e)}")
    
    def clear_all_preferences(self):
        """清除所有偏好"""
        try:
            # 五类偏好维度与 MemoryEntry.entry_type 对齐，全量清空
            preference_types = ['technician', 'time_period', 'duration', 'service', 'technician_type']
            for pref_type in preference_types:
                self.clear_preference(pref_type)
            self.logger.info("清除所有用户偏好")
        except Exception as e:
            self.logger.error(f"清除所有偏好失败: {str(e)}")
