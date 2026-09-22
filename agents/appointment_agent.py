from dotenv import load_dotenv
import uuid
from langchain_core.chat_history import InMemoryChatMessageHistory
from config.model_provider import create_chat_model
from config.constants import StateEnum
from agents.core.session import SessionContext, new_appointment_history
from .appointment import (
    InputParser,
    TechnicianFinder,
    AppointmentProcessor,
    MessageBuilder,
    AppointmentDatabase
)

load_dotenv()


class AppointmentAgent:
    """
    预约机器人主控制器

    职责：
    1. 初始化各个组件
    2. 协调整个预约流程（实例本身无状态，会话状态挂载在 SessionContext 上）
    """

    def __init__(self, session_id=None, unrelated_callback=None):
        # 基础设置（default_session_id 仅用于无会话的独立运行场景）
        self.default_session_id = session_id or str(uuid.uuid4())
        self.unrelated_callback = unrelated_callback

        # 初始化LLM
        self.llm = self._initialize_llm()

        # 初始化组件（全部无会话状态）
        self.input_parser = InputParser(self.llm)
        self.technician_finder = TechnicianFinder()
        self.message_builder = MessageBuilder()
        self.appointment_database = AppointmentDatabase()
        self.appointment_processor = AppointmentProcessor(
            self.input_parser,
            self.technician_finder,
            self.message_builder,
            self.appointment_database,
            self.llm
        )

    def _initialize_llm(self):
        """初始化通用聊天模型"""
        return create_chat_model(temperature=0)

    def _transient_session(self) -> SessionContext:
        """无会话场景下的临时上下文（独立运行/向后兼容）"""
        return SessionContext(
            session_id=self.default_session_id,
            chat_history=InMemoryChatMessageHistory()
        )

    async def run_stream(self, user_input=None, session: SessionContext = None,
                         enable_replan_signal: bool = False,
                         enable_confirm_signal: bool = False):
        """
        流式处理用户预约请求的主函数

        这是整个预约流程的入口点，协调各个组件完成预约。
        会话状态（预约历史、聊天历史、状态机）全部取自 session 参数。

        Args:
            enable_replan_signal: 图编排模式下为 True——档期冲突时发出
                [SIGNAL]replan_needed 交由 LangGraph replan 节点处理。
            enable_confirm_signal: 图编排模式下为 True——落库前确认与
                禁忌转人工发出 [SIGNAL]confirmation_pending / [SIGNAL]manual_review，
                交由 LangGraph confirm / manual_review 节点处理。
        """
        if user_input is None:
            user_input = input("用户：")

        ctx = session or self._transient_session()

        # 1. 解析用户输入（结构化输出 + 修复重试；内部 JSON 不向用户流式输出）
        extraction = await self.input_parser.extract(user_input, ctx.chat_history)

        try:
            # 2. 使用解析结果
            data = extraction.model_dump()
            finished = self.appointment_processor.update_history_from_data(
                ctx.appointment_history, data, user_input=user_input
            )

            # 3. 处理与预约无关的请求
            # 如果正在等待用户确认（推荐技师/预约摘要/禁忌回复），不要转交给归类机器人
            if (data.get("unrelated", False)
                    and not ctx.appointment_history.get('awaiting_confirmation')
                    and not ctx.appointment_history.get('awaiting_booking_confirmation')
                    and not ctx.appointment_history.get('contraindication_hit')):
                # 注意：这里不清空预约历史，保留用户已输入的信息
                # 只设置状态为CLASSIFY，让系统转交给其他机器人处理
                ctx.state.value = StateEnum.CLASSIFY

                async for token in self.appointment_processor.handle_unrelated_request(
                    user_input, self.unrelated_callback, ctx
                ):
                    yield token
                return

            # 4. 处理预约完成的情况
            if finished:
                recommendation_pending = False
                replan_signal = False
                confirmation_pending = False
                manual_review_signal = False
                self.appointment_processor.emit_replan_signal = enable_replan_signal
                self.appointment_processor.emit_confirm_signal = enable_confirm_signal
                try:
                    async for token in self.appointment_processor.handle_complete_appointment(
                        ctx.appointment_history, ctx.session_id
                    ):
                        # 检查是否有推荐等待确认
                        if token == "[SIGNAL]recommendation_pending":
                            recommendation_pending = True
                            continue
                        if token == "[SIGNAL]replan_needed":
                            # 冲突信号：保留预约槽位供 replan 节点读取，不重置
                            replan_signal = True
                            yield token
                            continue
                        if token == "[SIGNAL]confirmation_pending":
                            # 落库前确认：保留槽位供 confirm 节点输出摘要
                            confirmation_pending = True
                            yield token
                            continue
                        if token == "[SIGNAL]manual_review":
                            # 禁忌命中：保留命中信息供 manual_review 节点处理
                            manual_review_signal = True
                            yield token
                            continue
                        yield token
                finally:
                    self.appointment_processor.emit_replan_signal = False
                    self.appointment_processor.emit_confirm_signal = False

                # 只有在真正完成预约时才重置状态
                if (not recommendation_pending and not replan_signal
                        and not confirmation_pending and not manual_review_signal
                        and not ctx.appointment_history.get('awaiting_confirmation')
                        and not ctx.appointment_history.get('awaiting_booking_confirmation')):
                    self._reset_after_appointment(ctx)
                return

            # 5. 处理信息不完整/等待确认的情况
            if ctx.appointment_history.get('awaiting_booking_confirmation'):
                yield f"[REPLY][预约机器人]{self.message_builder.create_reconfirm_message()}"
                return
            async for token in self.appointment_processor.handle_incomplete_info(data, ctx.appointment_history):
                yield token

        except Exception as e:
            yield self.message_builder.create_parse_error_message()

    def _reset_after_appointment(self, ctx: SessionContext):
        """预约完成后重置会话内的预约状态"""
        ctx.appointment_history = new_appointment_history()
        ctx.chat_history.clear()
        ctx.state.value = StateEnum.CLASSIFY
