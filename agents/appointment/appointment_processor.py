"""
预约处理器

负责协调整个预约流程
"""

from typing import Dict, Any, AsyncGenerator
from .input_parser import InputParser
from .technician_finder import TechnicianFinder
from .message_builder import MessageBuilder
from .appointment_database import AppointmentDatabase
from .weather_tool import WeatherTool
from config.domain_terms import CONTRAINDICATION_TERMS
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate

# 确认/取消回复的关键词（B4 落库前确认 + 技师推荐确认共用）
POSITIVE_RESPONSES = ['是', '好', '可以', '同意', '确定', '确认', 'yes', 'ok', '行', '没问题']
NEGATIVE_RESPONSES = ['不', '不要', '不行', '不同意', '取消', '换', 'no', '算了']


class AppointmentProcessor:
    """预约处理器"""
    
    def __init__(self, input_parser: InputParser, technician_finder: TechnicianFinder,
                 message_builder: MessageBuilder, appointment_database: AppointmentDatabase, llm=None):
        self.input_parser = input_parser
        self.technician_finder = technician_finder
        self.message_builder = message_builder
        self.appointment_database = appointment_database
        self.llm = llm
        # 图编排模式下开启：档期冲突时向上发出 replan 信号（由 LangGraph replan 节点处理）
        self.emit_replan_signal = False
        # 图编排模式下开启：落库前确认/禁忌转人工信号（由 confirm/manual_review 节点处理）
        self.emit_confirm_signal = False
        
        # 初始化天气工具和 agent
        if self.llm:
            self.weather_tool = WeatherTool()
            self.tools = [self.weather_tool]
            
            # 创建 agent prompt
            self.agent_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个智能助手，可以获取天气信息并生成个性化的预约成功提示。"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            # 创建 agent
            self.weather_agent = create_openai_tools_agent(self.llm, self.tools, self.agent_prompt)
            self.agent_executor = AgentExecutor(agent=self.weather_agent, tools=self.tools, verbose=True)
    
    def update_history_from_data(self, appointment_history: Dict[str, Any], data: Dict[str, Any],
                                 user_input: str = None) -> bool:
        """从解析数据更新预约历史"""
        # B4：等待落库确认时，处理确认/取消/禁忌筛查回复
        if appointment_history.get('awaiting_booking_confirmation'):
            return self._handle_booking_confirmation(appointment_history, user_input or "", data)

        # 检查是否在等待用户确认推荐技师
        if appointment_history.get('awaiting_confirmation'):
            return self._handle_recommendation_response(appointment_history, data)
        
        # 只更新有值的字段，避免覆盖之前的信息
        for key in ["duration", "gender", "start_time", "project", "technician_name"]:
            if data.get(key) and data[key] != "未知":
                appointment_history[key] = data[key]
        
        # preference特殊处理
        if data.get("preference") and data["preference"] != "未知":
            appointment_history["preference"] = data["preference"]
        
        # 检查是否收集齐所有必需信息
        # 必需信息：时间、项目、时长
        # 如果指定了技师名，则不需要性别；否则性别也是必需的
        required_fields = ["start_time", "project", "duration"]
        technician_name = appointment_history.get("technician_name")
        
        if not technician_name or technician_name == "未知":
            # 没有指定技师，需要性别来筛选
            required_fields.append("gender")
        
        has_all_required = all(
            appointment_history.get(field) and appointment_history[field] != "未知" 
            for field in required_fields
        )
        
        # 如果信息完整，但是指定的技师不可用且需要推荐，则不认为预约"完成"
        if has_all_required and technician_name and technician_name != "未知":
            # 检查指定技师是否可用，如果不可用则进入推荐流程
            # 这个检查留到 handle_complete_appointment 中进行
            pass
        
        return has_all_required

    def _handle_recommendation_response(self, appointment_history: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """处理用户对推荐技师的回应"""
        user_response = data.get('confirmation', '').lower()

        # 判断用户是否同意推荐
        is_positive = any(pos in user_response for pos in POSITIVE_RESPONSES)
        is_negative = any(neg in user_response for neg in NEGATIVE_RESPONSES)
        
        if is_positive and not is_negative:
            # 用户同意推荐，更新技师信息
            recommended_tech = appointment_history.get('recommended_technician')
            if recommended_tech:
                appointment_history['confirmed_technician'] = recommended_tech
                appointment_history['awaiting_confirmation'] = False
                return True  # 表示可以进行预约
        elif is_negative:
            # 用户拒绝推荐
            appointment_history['recommendation_declined'] = True
            appointment_history['awaiting_confirmation'] = False
            return True  # 表示需要处理拒绝情况
        
        # 用户回应不明确，继续等待
        # 这里返回 False，表示信息还不完整，需要继续等待用户输入
        return False

    def _handle_booking_confirmation(self, appointment_history: Dict[str, Any],
                                     user_input: str, data: Dict[str, Any]) -> bool:
        """处理落库前确认与禁忌筛查回复（B4）

        Returns:
            True 表示进入完成处理（确认落库/取消/禁忌转人工），False 表示继续等待明确回复。
        """
        # 禁忌筛查：规则匹配用户原文（宁误拦不漏拦）
        hit_terms = [t for t in CONTRAINDICATION_TERMS if t in user_input]
        if hit_terms:
            appointment_history['contraindication_hit'] = hit_terms
            appointment_history['awaiting_booking_confirmation'] = False
            return True  # 交由 handle_complete_appointment 转人工

        user_response = data.get('confirmation', '').lower()
        is_positive = any(pos in user_response for pos in POSITIVE_RESPONSES)
        is_negative = any(neg in user_response for neg in NEGATIVE_RESPONSES)

        if is_positive and not is_negative:
            # 用户确认：标记落库（pending_technician 已存）
            appointment_history['booking_confirmed'] = True
            appointment_history['awaiting_booking_confirmation'] = False
            return True
        if is_negative:
            appointment_history['booking_cancelled'] = True
            appointment_history['awaiting_booking_confirmation'] = False
            return True

        # 回应不明确，继续等待
        return False
    
    async def handle_unrelated_request(self, user_input: str, unrelated_callback, session) -> AsyncGenerator[str, None]:
        """处理与预约无关的请求（session 为会话上下文）"""
        # 注意：这里不重置状态，因为在调用处已经设置了状态
        # 保持预约历史不被清空

        if unrelated_callback:
            try:
                yield "[REPLY][预约机器人]和预约信息无关，已交给归类机器人处理\n"
                result = await unrelated_callback(user_input, session)
                if hasattr(result, '__aiter__'):
                    async for token in result:
                        yield token
                else:
                    yield result
            except Exception as e:
                yield f"[ERROR]处理请求时发生错误: {str(e)}\n"
                yield self.message_builder.create_unrelated_message()
        else:
            yield self.message_builder.create_unrelated_message()
    
    async def handle_complete_appointment(self, appointment_history: Dict[str, Any],
                                        session_id: str) -> AsyncGenerator[str, None]:
        """处理预约信息完整的情况"""
        # B4：禁忌命中 → 转人工（不落库）
        if appointment_history.get('contraindication_hit'):
            if self.emit_confirm_signal:
                # 图编排模式：由 manual_review 节点输出转人工话术并重置会话
                yield "[SIGNAL]manual_review"
                return
            hit_terms = appointment_history.pop('contraindication_hit', [])
            appointment_history.pop('pending_technician', None)
            reply = self.message_builder.create_contraindication_referral_message(hit_terms)
            yield f"[REPLY][预约机器人]{reply}"
            return

        # B4：用户取消确认
        if appointment_history.get('booking_cancelled'):
            appointment_history.pop('booking_cancelled', None)
            appointment_history.pop('pending_technician', None)
            reply = self.message_builder.create_booking_cancelled_message()
            yield f"[REPLY][预约机器人]{reply}"
            return

        # B4：用户已确认预约摘要 → 落库
        if appointment_history.get('booking_confirmed'):
            tech = appointment_history.pop('pending_technician', None)
            appointment_history.pop('booking_confirmed', None)
            if tech:
                reply = await self._process_successful_appointment(tech, appointment_history, session_id)
                yield f"[REPLY][预约机器人]{reply}"
            return

        # 检查是否用户拒绝了推荐
        if appointment_history.get('recommendation_declined'):
            reply = self.message_builder.create_recommendation_declined_message(self.llm)
            yield f"[REPLY][预约机器人]{reply}"
            # 清理状态
            appointment_history.pop('recommendation_declined', None)
            appointment_history.pop('recommended_technician', None)
            appointment_history.pop('original_technician', None)
            return
        
        # 检查是否用户确认了推荐技师
        if appointment_history.get('confirmed_technician'):
            tech = appointment_history['confirmed_technician']
            # 标记为推荐技师用于成功消息显示
            tech['is_recommendation'] = True
            tech['original_technician'] = appointment_history.get('original_technician')
            reply = await self._process_successful_appointment(tech, appointment_history, session_id)
            yield f"[REPLY][预约机器人]{reply}"
            # 清理状态
            appointment_history.pop('confirmed_technician', None)
            appointment_history.pop('recommended_technician', None)
            appointment_history.pop('original_technician', None)
            return
        
        # 检查是否在等待用户确认推荐技师
        if appointment_history.get('awaiting_confirmation'):
            # 用户回应不明确，重新询问
            yield f"[REPLY][预约机器人]\n机器人：请您明确回复\"是\"或\"不\"，我好为您安排预约。\n"
            return
        
        # 收集思考过程
        thought_msgs = []
        def collect_thoughts(msg):
            thought_msgs.append(msg)
        
        tech = self.technician_finder.find_technician_with_thought(appointment_history, collect_thoughts)
        
        # 输出所有思考过程
        for msg in thought_msgs:
            yield msg
        
        technician_name = appointment_history.get("technician_name")
        
        if tech:
            # 检查是否是需要确认的推荐
            if tech.get('requires_confirmation'):
                original_tech = tech.get('original_technician')
                recommended_tech = tech.get('recommended_technician')
                
                # 生成推荐消息
                recommendation_msg = self.message_builder.create_technician_recommendation_message(
                    original_tech, recommended_tech, appointment_history, self.llm
                )
                yield f"[REPLY][预约机器人]{recommendation_msg}"
                
                # 将推荐信息存储在预约历史中，等待用户确认
                appointment_history['recommended_technician'] = recommended_tech
                appointment_history['original_technician'] = original_tech
                appointment_history['awaiting_confirmation'] = True
                
                # 重要：告诉调用方这个预约还没有真正完成，需要继续等待用户输入
                yield "[SIGNAL]recommendation_pending"
                return
            else:
                # B4：落库前确认——暂存理疗师并等待用户确认摘要
                appointment_history['pending_technician'] = tech
                appointment_history['awaiting_booking_confirmation'] = True
                if self.emit_confirm_signal:
                    # 图编排模式：由 confirm 节点输出摘要与禁忌筛查问询
                    yield "[SIGNAL]confirmation_pending"
                else:
                    reply = self.message_builder.create_booking_confirmation_message(tech, appointment_history)
                    yield f"[REPLY][预约机器人]{reply}"
                return
        else:
            if self.emit_replan_signal:
                # 图编排模式：档期冲突交由 replan 节点重规划（不直接输出失败消息）
                yield "[SIGNAL]replan_needed"
            else:
                reply = self.message_builder.create_appointment_failure_message(technician_name)
                yield f"[REPLY][预约机器人]{reply}"
    
    async def _process_successful_appointment(self, tech: Dict[str, Any], 
                                           appointment_history: Dict[str, Any], session_id: str) -> str:
        """处理预约成功的情况，并结合北京天气生成温馨提示"""
        start_time, end_time, duration_min = self.technician_finder.parse_time_and_duration(
            appointment_history["start_time"], 
            appointment_history["duration"]
        )
        # 保存预约到数据库
        success = self.appointment_database.save_appointment(
            tech["id"], start_time, end_time, appointment_history, session_id
        )
        if success:
            # 更新内存中的忙碌时段
            self.appointment_database.update_memory_schedule(tech["id"], start_time, end_time)
            # 使用 LLM agent 生成结合北京天气的温馨提示
            if self.llm and hasattr(self, 'agent_executor'):
                prompt = f"请获取北京今天的天气信息，然后结合天气情况为用户生成一段温馨的预约成功提示。技师姓名：{tech['name']}，性别：{tech['gender']}。请根据天气给出合适的建议和关怀。"
                try:
                    result = await self.agent_executor.ainvoke({"input": prompt})
                    agent_output = result.get("output", "")
                    return f"\n机器人：已为您预约技师：{tech['name']}，性别：{tech['gender']}。预约成功！\n{agent_output}\n"
                except Exception as e:
                    print(f"Agent调用失败: {e}")
                    return self.message_builder.create_appointment_success_message(tech)
            else:
                return self.message_builder.create_appointment_success_message(tech)
        else:
            return self.message_builder.create_save_failure_message()
    
    async def handle_incomplete_info(self, data: Dict[str, Any], appointment_history: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """处理信息不完整的情况"""
        # 确定缺失的信息
        missing = []
        technician_name = appointment_history.get("technician_name")
        
        # 基本必需信息
        if not appointment_history.get("start_time") or appointment_history.get("start_time") == "未知":
            missing.append("start_time")
        if not appointment_history.get("project") or appointment_history.get("project") == "未知":
            missing.append("project")
        if not appointment_history.get("duration") or appointment_history.get("duration") == "未知":
            missing.append("duration")
        
        # 如果没有指定技师名，则需要性别
        if not technician_name or technician_name == "未知":
            if not appointment_history.get("gender") or appointment_history.get("gender") == "未知":
                missing.append("gender")
        
        reply = self.message_builder.create_missing_info_questions(missing)
        yield f"[THOUGHT][预约机器人]用户的预约信息不完整，缺少：{', '.join(missing)}，我需要询问用户补充这些信息"
        yield f"[REPLY][预约机器人]{reply}"
