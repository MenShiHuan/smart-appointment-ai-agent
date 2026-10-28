"""
预约处理器

负责协调整个预约流程
"""

from typing import Dict, Any, AsyncGenerator, Optional
from .input_parser import InputParser
from .technician_finder import TechnicianFinder
from .message_builder import MessageBuilder
from .appointment_database import AppointmentDatabase
from .weather_tool import WeatherTool
from config.domain_terms import CONTRAINDICATION_TERMS

# 确认/取消回复的关键词（B4 落库前确认 + 技师推荐确认共用）
POSITIVE_RESPONSES = ['是', '好', '可以', '同意', '确定', '确认', 'yes', 'ok', '行', '没问题']
# "改/换一位/另约"表示变更意图（推荐确认语境下=拒绝原推荐）
NEGATIVE_RESPONSES = ['不', '不要', '不行', '不同意', '取消', '换', '改', '换一位', '另约', 'no', '算了']


class AppointmentProcessor:
    """预约处理器：协调 抽取 → 技师匹配 → 推荐/落库确认 → 落库 全流程

    以 [SIGNAL]xxx 协议 token 向上透出状态（replan_needed / confirmation_pending /
    manual_review / recommendation_pending），由调用方（图编排层）决定后续动作。
    """
    
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

        # 天气工具（门店城市配置化后直接调用，不经过 ReAct agent，避免英文/追问泄漏）
        self.weather_tool = WeatherTool()
    
    def update_history_from_data(self, appointment_history: Dict[str, Any], data: Dict[str, Any],
                                 user_input: str = None) -> bool:
        """从解析数据更新预约历史

        必填槽位：时间 + 项目（config/domain_terms.REQUIRED_SLOT_FIELDS）；
        时长/性别/指定理疗师/偏好均为选填，时长缺省按项目标准时长补齐。
        """
        # B4：等待落库确认时，处理确认/取消/禁忌筛查回复
        if appointment_history.get('awaiting_booking_confirmation'):
            return self._handle_booking_confirmation(appointment_history, user_input or "", data)

        # 检查是否在等待用户确认推荐技师
        if appointment_history.get('awaiting_confirmation'):
            return self._handle_recommendation_response(appointment_history, data, user_input)

        # 只更新有值的字段，避免覆盖之前的信息
        for key in ["duration", "gender", "start_time", "project", "technician_name"]:
            if data.get(key) and data[key] != "未知":
                appointment_history[key] = data[key]

        # preference特殊处理
        if data.get("preference") and data["preference"] != "未知":
            appointment_history["preference"] = data["preference"]

        # 必填信息判定：时间 + 项目
        from config.domain_terms import REQUIRED_SLOT_FIELDS, PROJECT_DEFAULT_DURATIONS, DEFAULT_DURATION_MINUTES
        has_all_required = all(
            appointment_history.get(field) and appointment_history[field] != "未知"
            for field in REQUIRED_SLOT_FIELDS
        )

        # 时长选填：信息齐全且用户未提时长时，按项目标准时长补齐
        if has_all_required and not appointment_history.get("duration"):
            project = appointment_history.get("project", "")
            appointment_history["duration"] = (
                f"{PROJECT_DEFAULT_DURATIONS.get(project, DEFAULT_DURATION_MINUTES)}分钟")

        return has_all_required

    def _handle_recommendation_response(self, appointment_history: Dict[str, Any],
                                        data: Dict[str, Any],
                                        user_input: Optional[str] = None) -> bool:
        """处理用户对推荐技师的回应

        True=同意推荐（确认技师）或拒绝但携带新选择（继续正常匹配）；
        False=回应不明确，继续等待；纯拒绝时置 recommendation_declined 标记。
        """
        user_response = (data.get('confirmation') or '').lower()
        # 兜底：confirmation 未被抽取（未知/空）时，用用户原文做肯定/否定关键词判定，
        # 覆盖"我不换别人，那我改到后天下午三点吧"这类含否定/变更意图的回复
        if not user_response or user_response == "未知":
            user_response = (user_input or '').lower()

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
            appointment_history['awaiting_confirmation'] = False

            # 拒绝时同句给出新选择（换技师/改时间等）：合并新槽位并清理推荐状态，
            # 继续走正常匹配链路（e2e-002/003 场景）
            new_slots = {k: data[k] for k in
                         ("duration", "gender", "start_time", "project", "technician_name", "preference")
                         if data.get(k) and data[k] != "未知"}
            if new_slots:
                appointment_history.pop('recommended_technician', None)
                appointment_history.pop('original_technician', None)
                appointment_history.update(new_slots)
                return True

            # 纯拒绝（无新选择）：走拒绝话术分支
            appointment_history['recommendation_declined'] = True
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
                # 兜底：回调失败时透出错误并回退无关请求话术，保证至少一条有效回复
                yield f"[ERROR]处理请求时发生错误: {str(e)}\n"
                yield self.message_builder.create_unrelated_message()
        else:
            yield self.message_builder.create_unrelated_message()
    
    async def handle_complete_appointment(self, appointment_history: Dict[str, Any],
                                        session_id: str) -> AsyncGenerator[str, None]:
        """处理预约信息完整的情况

        分支按优先级依次判定：禁忌转人工 > 用户取消 > 已确认落库 > 拒绝推荐 >
        确认推荐落库 > 等待明确回复 > 技师检索（推荐待确认 / 落库前确认 /
        档期冲突通过 [SIGNAL] 协议向上透出）。
        """
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
        """处理预约成功的情况，并结合当地天气生成温馨提示（理疗后保暖/防晒建议）"""
        start_time, end_time, duration_min = self.technician_finder.parse_time_and_duration(
            appointment_history["start_time"],
            appointment_history["duration"]
        )
        # 保存预约到数据库
        success = self.appointment_database.save_appointment(
            tech["id"], start_time, end_time, appointment_history, session_id
        )
        if success:
            # 天气温馨提示：门店城市配置化（config/domain_terms.STORE_CITY），
            # 工具直调 + 单次 LLM 生成，输出约束为中文、不追问、不泄漏英文
            if self.llm and self.weather_tool:
                try:
                    from config.domain_terms import STORE_CITY
                    weather_text = await self.weather_tool._arun(STORE_CITY or "")
                    compose_prompt = (
                        f"请根据以下天气信息，为预约成功的用户写一句中文温馨提示"
                        f"（理疗后保暖/防晒等建议，40字以内）。\n"
                        f"要求：只输出中文建议本身，不要提问、不要使用英文、不要重复天气数据。\n"
                        f"天气信息：{weather_text}\n"
                        f"理疗师：{tech['name']}（{tech['gender']}）")
                    response = await self.llm.ainvoke(
                        [{"role": "user", "content": compose_prompt}])
                    tip = (response.content if hasattr(response, "content")
                           else str(response)).strip()
                    if tip:
                        return (f"\n机器人：已为您预约理疗师：{tech['name']}，"
                                f"性别：{tech['gender']}。预约成功！\n{tip}\n")
                except Exception as e:
                    print(f"天气温馨提示生成失败，使用默认话术: {e}")
            return self.message_builder.create_appointment_success_message(tech)
        else:
            return self.message_builder.create_save_failure_message()
    
    async def handle_incomplete_info(self, data: Dict[str, Any], appointment_history: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """处理信息不完整的情况（必填槽位：时间 + 项目）"""
        from config.domain_terms import REQUIRED_SLOT_FIELDS
        missing = [
            field for field in REQUIRED_SLOT_FIELDS
            if not appointment_history.get(field) or appointment_history.get(field) == "未知"
        ]

        # 兜底：槽位齐全却走到这里（异常路径），转重新确认而非输出"缺少："空话
        if not missing:
            yield f"[REPLY][预约机器人]{self.message_builder.create_reconfirm_message()}"
            return

        reply = self.message_builder.create_missing_info_questions(missing)
        yield f"[THOUGHT][预约机器人]用户的预约信息不完整，缺少：{', '.join(missing)}，我需要询问用户补充这些信息"
        yield f"[REPLY][预约机器人]{reply}"
