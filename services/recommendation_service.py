"""
推荐调度服务

职责：
1. 定时生成用户行为推荐
2. 管理推荐调度任务
3. 提供手动触发推荐功能
"""

import asyncio
import schedule
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RecommendationService:
    """推荐调度服务类"""
    # 后台定时调度入口：每小时检查离线记忆整合触发条件（C4a），并提供手动触发与状态查询
    
    def __init__(self):
        # 延迟导入避免循环依赖
        self._behavior_agent = None
        self.is_running = False
        self.scheduler_thread = None
    
    @property
    def behavior_agent(self):
        """懒加载用户行为服务"""
        if self._behavior_agent is None:
            from services.user_behavior_service import UserBehaviorService
            self._behavior_agent = UserBehaviorService()
        return self._behavior_agent
    
    def generate_recommendations_job(self) -> Optional[List[Dict[str, Any]]]:
        """定时生成推荐的任务"""
        try:
            logger.info("开始执行定时推荐生成任务...")
            # 通过用户行为服务分析用户模式并生成推荐
            # TODO: 实现基于用户行为的推荐逻辑
            recommendations = []

            if recommendations:
                logger.info(f"成功生成 {len(recommendations)} 条推荐:")
                for rec in recommendations:
                    logger.info(f"- {rec['type']}: {rec['content'][:50]}...")
                return recommendations
            else:
                logger.info("本次没有生成新的推荐")
                return None

        except Exception as e:
            logger.error(f"定时推荐生成任务失败: {str(e)}")
            return None

    # 记忆服务进程级缓存：避免每次任务重复创建 engine 与 LLM 客户端
    _memory_service = None

    def run_memory_integration_job(self) -> Optional[Dict[str, Any]]:
        """离线记忆整合调度任务（C4a）：触发判定由 MemoryService 内部完成

        MemoryService 进程级复用（每次新建会重复建 engine 与 LLM 客户端）。
        """
        try:
            from services.memory_service import MemoryService
            if RecommendationService._memory_service is None:
                RecommendationService._memory_service = MemoryService()
            logger.info("检查离线记忆整合触发条件...")
            report = RecommendationService._memory_service.run_integration_job()
            if report.get("triggered"):
                logger.info(f"记忆整合完成: 处理行为 {report['processed_behaviors']} 条，"
                            f"新建 {report['created']} / 合并 {report['merged']} / "
                            f"降权 {report['downweighted']} / 退役 {report['retired']} / "
                            f"固化 {report['promoted']}")
            else:
                logger.info(f"记忆整合未触发: {report.get('reason')}")
            return report
        except Exception as e:
            logger.error(f"离线记忆整合任务失败: {str(e)}")
            return None
    
    def start_scheduler(self) -> bool:
        """启动定时任务调度器"""
        if self.is_running:
            logger.warning("调度器已经在运行中")
            return False
        
        try:
            # 每小时检查离线记忆整合触发条件（C4a）
            schedule.every().hour.do(self.run_memory_integration_job)
            
            self.is_running = True
            
            def run_scheduler():
                logger.info("推荐调度器已启动")
                while self.is_running:
                    schedule.run_pending()
                    time.sleep(60)  # 每分钟检查一次
                logger.info("推荐调度器已停止")
            
            # 在后台线程中运行调度器
            self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            self.scheduler_thread.start()
            return True
            
        except Exception as e:
            logger.error(f"启动推荐调度器失败: {str(e)}")
            return False
        
    def stop_scheduler(self) -> bool:
        """停止定时任务调度器"""
        try:
            self.is_running = False
            schedule.clear()
            logger.info("推荐调度器已停止")
            return True
        except Exception as e:
            logger.error(f"停止推荐调度器失败: {str(e)}")
            return False
    
    def run_immediate_check(self) -> Optional[List[Dict[str, Any]]]:
        """立即执行一次推荐检查（用于测试或手动触发）"""
        logger.info("执行立即推荐检查...")
        return self.generate_recommendations_job()
    
    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            "is_running": self.is_running,
            "thread_alive": self.scheduler_thread.is_alive() if self.scheduler_thread else False,
            "next_job": str(schedule.next_run()) if schedule.jobs else None,
            "total_jobs": len(schedule.jobs)
        }

# 测试用的手动运行函数
if __name__ == "__main__":
    print("启动推荐调度器测试...")
    service = RecommendationService()
    service.start_scheduler()
    
    try:
        # 运行10分钟用于测试
        time.sleep(600)
    except KeyboardInterrupt:
        print("收到中断信号，停止调度器...")
    finally:
        service.stop_scheduler()
