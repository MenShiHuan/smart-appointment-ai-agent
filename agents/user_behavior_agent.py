"""
简化版用户行为分析代理

核心功能：
1. 分析用户偏好
2. 判断回访时机
3. 生成回访消息
"""

import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from config.model_provider import create_chat_model
from .user_behavior import PatternAnalyzer, BehaviorRecorder, PreferenceManager

load_dotenv()


class UserBehaviorAgent:
    """简化版用户行为分析代理"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 延迟导入Services避免循环依赖
        self._user_behavior_service = None
        self._technician_service = None
        
        # 初始化LLM - 按照其他agent的模式
        self.llm = self._initialize_llm()
        
        # 通过 Services 层访问数据，遵守五层架构约束（Agents 层禁止直连 DB）
        from services.user_behavior_service import UserBehaviorService
        self.behavior_service = UserBehaviorService()

        # 为保持兼容性，创建适配器版本的组件
        self.pattern_analyzer = PatternAnalyzer(self.behavior_service)
        self.behavior_recorder = BehaviorRecorder(self.behavior_service)
        self.preference_manager = PreferenceManager(self.behavior_service)
    
    @property
    def user_behavior_service(self):
        """懒加载用户行为服务"""
        if self._user_behavior_service is None:
            from services.user_behavior_service import UserBehaviorService
            self._user_behavior_service = UserBehaviorService()
        return self._user_behavior_service

    @property
    def technician_service(self):
        """懒加载技师服务（通过 Services 层访问）"""
        if self._technician_service is None:
            from services.technician_service import TechnicianService
            self._technician_service = TechnicianService()
        return self._technician_service

    def _initialize_llm(self):
        """初始化通用聊天模型"""
        return create_chat_model(temperature=0.7)
    
    def record_behavior(self, action_type: str, action_data: Dict[str, Any], 
                       technician_id: str = None, session_id: str = "default_session") -> bool:
        """记录用户行为"""
        try:
            # 优先使用Services层，保持向后兼容
            return self.user_behavior_service.record_behavior(
                user_id="default_user",  # 统一使用default_user作为用户ID
                action_type=action_type,
                action_data=action_data,
                technician_id=technician_id,
                session_id=session_id
            )
        except Exception as e:
            self.logger.error(f"记录用户行为失败: {str(e)}")
            # 回退到原有方法
            try:
                return self.behavior_recorder.record_behavior(
                    action_type=action_type,
                    action_data=action_data,
                    technician_id=technician_id,
                    session_id=session_id
                )
            except Exception as fallback_error:
                self.logger.error(f"回退方法也失败: {str(fallback_error)}")
                return False
    
    def get_user_analysis(self, user_id: str = "default_user") -> Optional[Dict[str, Any]]:
        """获取用户分析数据"""
        try:
            preferences = self.pattern_analyzer.analyze_user_preferences(user_id)
            if not preferences:
                return None
            
            # 计算距离上次预约的天数
            last_appointment = preferences.get('last_appointment_date')
            days_since_last = None
            if last_appointment:
                from datetime import datetime
                if isinstance(last_appointment, str):
                    last_appointment = datetime.fromisoformat(last_appointment.replace('Z', '+00:00'))
                days_since_last = (datetime.now() - last_appointment).days
            
            return {
                'favorite_technician_id': preferences.get('favorite_technician_id'),
                'favorite_service': preferences.get('favorite_service'),
                'favorite_duration': preferences.get('favorite_duration'),
                'total_appointments': preferences.get('total_appointments'),
                'days_since_last_appointment': days_since_last,
                'should_send_reminder': self.pattern_analyzer.should_send_return_reminder(user_id)
            }
        except Exception as e:
            self.logger.error(f"获取用户分析失败: {str(e)}")
            return None
    
    def generate_reminder_message(self, user_id: str = "default_user") -> Optional[str]:
        """生成回访提醒消息"""
        try:
            return self.pattern_analyzer.generate_return_message(user_id)
        except Exception as e:
            self.logger.error(f"生成提醒消息失败: {str(e)}")
            return None

    async def answer_preference_query_stream(self, user_input: str, user_id: str = "default_user"):
        """
        回答用户关于自身历史预约与偏好的查询（长记忆召回，流式）

        场景：用户问"我上次预约的是哪位理疗师"、"我常做什么项目"等，
        由分类器路由到本 Agent（memory 分类），从长期偏好画像生成回答。
        """
        try:
            analysis = self.get_user_analysis(user_id)

            # 无任何偏好数据时给出引导性回答
            if not analysis or (
                not analysis.get('favorite_technician_id')
                and not analysis.get('favorite_service')
            ):
                fallback = ("您目前还没有预约记录。完成首次预约后，"
                            "我就能帮您记住偏好的理疗师和服务项目。")
                yield "[THOUGHT][记忆机器人] 记忆机器人：我来查看您的历史偏好记录...\n"
                yield "[REPLY][记忆机器人]"
                for char in fallback:
                    yield char
                return

            # 组装用户画像信息
            tech_name = '您偏爱的理疗师'
            tech_strength = ''
            if analysis.get('favorite_technician_id'):
                tech_info = self.technician_service.get_technician_by_id(
                    analysis['favorite_technician_id']
                )
                if tech_info:
                    tech_name = tech_info.get('name', tech_name)
                    tech_strength = tech_info.get('strength', '')

            prompt = f"""用户向你询问自己的历史偏好，请根据用户画像回答。

用户画像：
- 偏好的理疗师：{tech_name}{'（擅长：' + tech_strength + '）' if tech_strength else ''}
- 常用服务项目：{analysis.get('favorite_service', '未知')}
- 常用时长：{analysis.get('favorite_duration', '未知')}分钟
- 累计预约次数：{analysis.get('total_appointments', 0)}
- 距上次预约：{analysis.get('days_since_last_appointment', '未知')}天

用户的问题：{user_input}

要求：
1. 用第二人称亲切自然地回答
2. 如实引用画像数据，没有的信息不要编造
3. 控制在60字以内"""

            yield "[THOUGHT][记忆机器人] 记忆机器人：我来查看您的历史偏好记录...\n"
            yield "[REPLY][记忆机器人]"
            async for chunk in self.llm.astream([{"role": "user", "content": prompt}]):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    yield content

        except Exception as e:
            self.logger.error(f"记忆查询回答失败: {type(e).__name__}: {str(e)}")
            yield "[REPLY][记忆机器人]抱歉，我暂时想不起您的偏好记录，请稍后再试。"

    async def generate_personalized_reminder(self, user_id: str = "default_user", 
                                           available_times: list = None) -> Optional[str]:
        """使用LLM生成个性化回访提醒消息"""
        try:
            self.logger.info(f"开始生成个性化提醒，用户ID: {user_id}")
            self.logger.info(f"可用时间: {available_times}")
            
            # 获取用户分析数据
            analysis = self.get_user_analysis(user_id)
            if not analysis or not analysis.get('favorite_technician_id'):
                self.logger.info("没有找到用户偏好数据，使用默认消息")
                return "尊敬的Tom，您好！好久没见了，要不要预约一个按摩放松一下？"
            
            self.logger.info(f"用户分析数据: {analysis}")
            
            # 获取技师信息（通过 Services 层，遵守分层架构）
            tech_info = self.technician_service.get_technician_by_id(analysis['favorite_technician_id'])
            
            tech_name = tech_info.get('name', '您偏爱的技师') if tech_info else '您偏爱的技师'
            tech_strength = tech_info.get('strength', '') if tech_info else ''
            service = analysis.get('favorite_service', '按摩')
            duration = analysis.get('favorite_duration', 60)
            
            self.logger.info(f"技师信息: {tech_name}, 特长: {tech_strength}")
            
            # 格式化可用时间
            times_text = "、".join([t["formatted"] for t in (available_times or [])[:3]]) if available_times else "暂时没有空闲时间"
            self.logger.info(f"格式化后的时间: {times_text}")
            
            # 构建LLM提示
            prompt = f"""请为健康理疗店生成一条温暖的回访消息。

客户信息：
- 称呼：尊敬的Tom
- 最喜欢的理疗师：{tech_name}
- 理疗师特长：{tech_strength}
- 常用服务：{service.replace('按摩', '理疗')}
- 常用时长：{duration}分钟
- 理疗师空闲时间：{times_text}

要求：
1. 语气亲切温暖，像老朋友一样
2. 提到理疗师的名字和特长
3. 结合客户的使用习惯
4. 如果有空闲时间，自然地提及具体时间
5. 最后邀请客户预约
6. 控制在80字以内
7. 直接输出消息内容，不要任何标记"""
            
            self.logger.info(f"准备调用LLM，提示内容长度: {len(prompt)}")
            
            # 调用LLM生成个性化消息
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            generated_message = response.content.strip()
            
            self.logger.info(f"LLM生成的消息: {generated_message}")
            
            return generated_message
            
        except Exception as e:
            self.logger.error(f"LLM生成个性化提醒失败: {type(e).__name__}: {str(e)}")
            # 回退到传统方法
            return self.generate_reminder_message(user_id)

    async def get_reminder_with_schedule(self, user_id: str = "default_user") -> Dict[str, Any]:
        """获取包含时间安排的完整提醒信息"""
        try:
            # 获取用户分析数据
            analysis = self.get_user_analysis(user_id)
            if not analysis or not analysis.get('favorite_technician_id'):
                message = await self.generate_personalized_reminder(user_id, [])
                return {
                    "message": message,
                    "technician_available_times": []
                }
            
            tech_id = analysis['favorite_technician_id']
            duration = analysis.get('favorite_duration', 60)
            
            # 查询技师的空闲时间（通过 Services 层，遵守分层架构）
            from datetime import datetime, timedelta

            technician_service = self.technician_service
            today = datetime.now()
            available_times = []
            
            # 检查今天从当前时间到晚上22点的空闲时段
            current_hour = datetime.now().hour
            start_hour = max(current_hour + 1, 9)  # 从下一小时开始，但不早于9点
            
            for hour in range(start_hour, 22):  # 9点到22点营业时间
                check_time = today.replace(hour=hour, minute=0, second=0, microsecond=0)
                end_time = check_time + timedelta(minutes=duration)
                
                if technician_service.is_technician_available(tech_id, check_time, end_time):
                    available_times.append({
                        "date": today.strftime("%Y-%m-%d"),
                        "time": f"{hour}:00",
                        "formatted": f"今天{hour}:00"
                    })
            
            # 如果今天没有空闲时间，检查明天
            if not available_times:
                tomorrow = today + timedelta(days=1)
                for hour in range(9, 22):  # 9点到22点营业时间
                    check_time = tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
                    end_time = check_time + timedelta(minutes=duration)
                    
                    if technician_service.is_technician_available(tech_id, check_time, end_time):
                        available_times.append({
                            "date": tomorrow.strftime("%Y-%m-%d"),
                            "time": f"{hour}:00",
                            "formatted": f"明天{hour}:00"
                        })
                        if len(available_times) >= 3:  # 最多显示3个时间
                            break
            
            # 生成个性化消息
            message = await self.generate_personalized_reminder(user_id, available_times)
            
            return {
                "message": message,
                "technician_available_times": available_times
            }
            
        except Exception as e:
            self.logger.error(f"获取提醒信息失败: {type(e).__name__}: {str(e)}")
            return {
                "message": "尊敬的Tom，您好！系统暂时无法查询技师时间，请稍后再试或直接联系我们预约。",
                "technician_available_times": []
            }
