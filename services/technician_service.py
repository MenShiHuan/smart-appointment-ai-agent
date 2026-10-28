"""技师服务：默认技师种子初始化、种子档案定向迁移（E1 配套）与排班查询封装"""
# utils/ai/technician_service.py

from typing import List, Dict, Any
from db.db_router import DatabaseRouter
import logging

logger = logging.getLogger(__name__)

class TechnicianService:
    """技师服务类 - 管理技师数据和默认初始化"""
    
    def __init__(self):
        self.db = DatabaseRouter()
        
        # 默认理疗师数据（10人，其中有两位擅长内容接近）
        self.default_technicians = [
            {
                "name": "张伟",
                "gender": "男",
                "strength": "擅长肩颈腰背理疗，手法力度大，善于缓解深层肌肉酸痛，注重肌肉深层放松"
            },
            {
                "name": "王强",
                "gender": "男",
                "strength": "深层肌肉理疗专家，手法扎实，专注于运动损伤康复和肌肉放松"
            },
            {
                "name": "李娜",
                "gender": "女",
                "strength": "从业多年经验丰富，手法专业细腻、深受顾客好评，擅长舒缓放松类理疗与肩颈调理，适合压力大、睡眠差人群"
            },
            {
                "name": "赵敏",
                "gender": "女",
                "strength": "精通经络理疗，善于调理亚健康，手法力度适中"
            },
            {
                "name": "刘洋",
                "gender": "男",
                "strength": "全身放松理疗专家，拉伸到位，适合喜欢全身放松的客户"
            },
            {
                "name": "孙丽",
                "gender": "女",
                "strength": "芳香精油理疗，舒缓情绪，适合女性客户"
            },
            {
                "name": "周杰",
                "gender": "男",
                "strength": "中医理疗，针对颈椎、腰椎问题有丰富经验"
            },
            {
                "name": "吴婷",
                "gender": "女",
                "strength": "头部理疗和足部理疗专家，助眠效果好"
            },
            {
                "name": "郑斌",
                "gender": "男",
                "strength": "手法力度大，适合喜欢重手法的客户，善于肌肉放松"
            },
            {
                "name": "何静",
                "gender": "女",
                "strength": "淋巴引流、面部护理，适合美容养生需求"
            }
        ]

    # 旧种子专长 → 新专长的定向迁移（E1 配套：补齐理疗师档案信息，供咨询链引用）
    _strength_migrations = {
        "李娜": ("手法细腻，擅长舒缓放松类理疗，适合压力大、睡眠差人群",
                 "从业多年经验丰富，手法专业细腻、深受顾客好评，擅长舒缓放松类理疗与肩颈调理，适合压力大、睡眠差人群"),
    }

    def _migrate_seed_strengths(self):
        """已有库的种子专长定向更新（仅当当前值恰为旧种子值时更新，不动运营修改）"""
        for name, (old, new) in self._strength_migrations.items():
            tech = self.db.technicians.get_technician_by_name(name)
            if tech and tech.get("strength") == old:
                self.db.technicians.update_technician(tech["id"], strength=new)
                logger.info(f"理疗师 {name} 专长档案已更新")

    def initialize_default_technicians(self) -> bool:
        """初始化默认技师数据"""
        # 库中已有技师时不做覆盖，仅执行种子专长迁移检查
        try:
            # 检查是否已有技师数据
            existing_technicians = self.db.technicians.get_all_technicians()

            if existing_technicians:
                logger.info(f"数据库中已有 {len(existing_technicians)} 位理疗师，执行种子档案迁移检查")
                self._migrate_seed_strengths()
                return True
            
            logger.info("数据库中无技师数据，开始初始化默认技师")
            
            # 添加默认技师
            for tech_data in self.default_technicians:
                try:
                    tech_id = self.db.technicians.add_technician(
                        name=tech_data['name'],
                        gender=tech_data['gender'],
                        strength=tech_data['strength']
                    )
                    logger.debug(f"添加技师: {tech_data['name']} (ID: {tech_id})")
                    
                except Exception as e:
                    logger.error(f"添加技师 {tech_data['name']} 失败: {e}")
                    return False
            
            # 验证初始化结果
            final_count = len(self.db.technicians.get_all_technicians())
            logger.info(f"技师初始化完成，共添加 {final_count} 位技师")
            return True
            
        except Exception as e:
            logger.error(f"技师初始化失败: {e}")
            return False

    def get_all_technicians(self) -> List[Dict[str, Any]]:
        """获取所有技师信息"""
        return self.db.technicians.get_all_technicians()

    def get_technician_by_name(self, name: str) -> Dict[str, Any]:
        """根据姓名获取技师信息"""
        return self.db.technicians.get_technician_by_name(name)

    def get_technician_by_id(self, technician_id: int) -> Dict[str, Any]:
        """根据ID获取技师信息"""
        return self.db.technicians.get_technician_by_id(technician_id)

    def get_technician_schedules(self, technician_id: int, date) -> List[Dict[str, Any]]:
        """获取技师指定日期的排班信息"""
        return self.db.technicians.get_technician_schedules(technician_id, date)

    def is_technician_available(self, technician_id: int, start_time, end_time) -> bool:
        """检查技师在指定时间段是否可用"""
        return self.db.technicians.is_technician_available(technician_id, start_time, end_time)

    def add_technician(self, name: str, gender: str = None, strength: str = None) -> int:
        """添加新技师"""
        return self.db.technicians.add_technician(name, gender, strength)

    def get_technicians_count(self) -> int:
        """获取技师总数"""
        technicians = self.db.technicians.get_all_technicians()
        return len(technicians)
