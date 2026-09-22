"""
用户输入解析器

负责解析用户输入并提取预约相关信息。
主路径：LLM 结构化输出（Pydantic with_structured_output）；
失败时带错误信息修复重试 1 次，再失败降级为默认值并记录 parse_error 计数。
"""
import json
from typing import Any, Dict, Generator

from langchain.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from .schemas import AppointmentExtraction


class InputParser:
    """用户输入解析器"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = self._create_prompt_template()

        # 结构化输出（provider 不支持时自动回退 prompt+JSON 模式）
        self._structured_llm = None
        try:
            self._structured_llm = self.llm.with_structured_output(AppointmentExtraction)
        except Exception as e:
            print(f"[InputParser] with_structured_output 不可用，使用 prompt+JSON 回退: {e}")

        # 解析失败计数（降级时 +1，供可观测性使用）
        self.parse_error_count = 0

    def _create_prompt_template(self) -> PromptTemplate:
        """创建预约信息提取的Prompt模板"""
        from config.time_config import time_config
        current_date = time_config.current_date_str()
        current_datetime = time_config.current_datetime_str()

        return PromptTemplate(
            input_variables=["history", "user_input"],
            template=(
                "你是一个预约机器人，负责帮用户预约服务。\n"
                f"当前日期是{current_date}，当前北京时间是{current_datetime}。\n"
                "当前已知信息：{history}\n"
                "用户输入：{user_input}\n"
                "特别注意：如果用户输入是对推荐技师确认问题的回应（如\"是\"、\"好\"、\"可以\"、\"不\"、\"不要\"等简短回复），请优先识别为confirmation，而不要标记为unrelated。\n"
                "重要：请你只输出纯JSON格式，不要添加任何markdown标记如```json或```，不要添加任何其他文字说明，直接输出JSON：\n"
                "{{\n"
                '  "gender": "技师性别（如男/女/未知）",\n'
                '  "start_time": "预约起始时间，必须转换为标准格式YYYY-MM-DD HH:MM。如果用户说今天下午3点，转换为当前日期 15:00；如果说明天上午10点，转换为明天日期 10:00。如果只说时间没说日期，默认为今天。如果完全没有时间信息则为未知",\n'
                '  "duration": "服务时长，统一转换为分钟数格式，如180分钟、60分钟。如果没有明确时长则为未知",\n'
                '  "project": "服务项目（如按摩/未知）",\n'
                '  "preference": "用户倾向（如力气大/力气小/无）",\n'
                '  "technician_name": "指定技师姓名（如果用户明确提到技师名字，如张伟、李小美等，否则为未知）",\n'
                '  "confirmation": "如果用户在回应技师推荐的确认问题，提取用户的回复内容（如是/好/可以/不/不要等），否则为未知",\n'
                '  "info_complete": "根据实际情况判断：1)如果指定了技师名且不为未知，需要start_time、project、duration都不为未知；2)如果没指定技师名，需要start_time、project、duration、gender都不为未知",\n'
                '  "unrelated": "如果用户的问题和预约无关（如问天气、聊天等），则为true，否则为false。注意：对推荐技师的确认回复（是/不等）不应标记为unrelated",\n'
                '  "missing_info": "如果info_complete为false，请列出缺少的关键信息，如[start_time, project]等"\n'
                "}}\n"
                "判断逻辑：\n"
                "1. 如果用户明确指定了技师姓名（如\"张伟技师\"、\"预约李小美\"等），请务必提取technician_name\n"
                "2. 如果用户在回应推荐技师的确认问题（如回复\"是\"、\"好\"、\"可以\"、\"不\"、\"不要\"等），请提取到confirmation字段，并且不要将其标记为unrelated\n"
                "3. 必需信息判断：\n"
                "   - 如果指定了技师名：需要start_time、project、duration\n"
                "   - 如果没指定技师名：需要start_time、project、duration、gender\n"
                "3. 只有当所有必需信息都不是'未知'时，info_complete才为true\n"
                "4. 如果用户的问题和预约无关，请将unrelated设为true\n"
                "再次强调：只输出纯JSON，不要有任何代码块标记或其他文字。"
            )
        )

    # ===========================================
    # 主入口：结构化提取（结构化输出 → 修复重试 → 降级兜底）
    # ===========================================

    async def extract(self, user_input: str, chat_history: InMemoryChatMessageHistory) -> AppointmentExtraction:
        """提取预约信息"""
        # 添加用户消息到历史
        chat_history.add_message(HumanMessage(content=user_input))

        history_str = self._build_history_str(chat_history)

        # 1. 结构化输出
        if self._structured_llm is not None:
            try:
                prompt_text = self.prompt.format(history=history_str, user_input=user_input)
                extraction = await self._structured_llm.ainvoke(prompt_text)
                chat_history.add_message(AIMessage(content=extraction.model_dump_json()))
                return extraction
            except Exception as e:
                print(f"[InputParser] 结构化输出失败，进入修复重试: {e}")

        # 2. prompt+JSON 路径（含一次修复重试）
        return await self._extract_with_repair(user_input, history_str, chat_history)

    async def _extract_with_repair(self, user_input: str, history_str: str,
                                   chat_history: InMemoryChatMessageHistory) -> AppointmentExtraction:
        """prompt+JSON 提取：首次失败时带错误信息重试 1 次"""
        raw = await self._invoke_raw(self.prompt.format(history=history_str, user_input=user_input))

        try:
            extraction = self._parse_raw(raw)
            chat_history.add_message(AIMessage(content=extraction.model_dump_json()))
            return extraction
        except Exception as first_error:
            print(f"[InputParser] 首次解析失败: {first_error}")

            # 修复重试：将原始输出与错误反馈给模型
            repair_prompt = (
                self.prompt.format(history=history_str, user_input=user_input)
                + f"\n你上次的输出无法解析：{raw[:200]}\n错误信息：{first_error}\n请直接输出纯JSON。"
            )
            raw_retry = await self._invoke_raw(repair_prompt)
            try:
                extraction = self._parse_raw(raw_retry)
                chat_history.add_message(AIMessage(content=extraction.model_dump_json()))
                return extraction
            except Exception as retry_error:
                print(f"[InputParser] 修复重试仍失败，降级为默认值: {retry_error}")
                self.parse_error_count += 1
                chat_history.add_message(AIMessage(content=raw_retry))
                return self._default_extraction()

    async def _invoke_raw(self, prompt_text: str) -> str:
        """调用 LLM 获取原始文本输出"""
        response = await self.llm.ainvoke([{"role": "user", "content": prompt_text}])
        return response.content if hasattr(response, "content") else str(response)

    def _parse_raw(self, raw: str) -> AppointmentExtraction:
        """解析原始 JSON 并校验为 Pydantic 模型"""
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("输出不是 JSON 对象")
        return AppointmentExtraction.model_validate(data)

    def _default_extraction(self) -> AppointmentExtraction:
        """解析彻底失败时的默认值（与原降级行为一致）"""
        return AppointmentExtraction(missing_info=["所有信息"])

    def _build_history_str(self, chat_history) -> str:
        """构建历史字符串"""
        return "\n".join(
            [f"用户：{m.content}" if m.type == "human" else f"机器人：{m.content}"
             for m in chat_history.messages]
        )

    # ===========================================
    # 兼容旧接口（prompt+JSON 流式路径）
    # ===========================================

    def parse_stream(self, user_input: str, chat_history: InMemoryChatMessageHistory) -> Generator[str, None, str]:
        """流式解析用户输入（兼容旧接口）"""
        # 添加用户消息到历史
        chat_history.add_message(HumanMessage(content=user_input))

        # 构建历史字符串
        history_str = self._build_history_str(chat_history)

        # 流式调用LLM
        response_stream = (self.prompt | self.llm).stream({"history": history_str, "user_input": user_input})
        ai_content = ""

        for chunk in response_stream:
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            ai_content += token
            yield token

        # 添加AI回复到历史
        chat_history.add_message(AIMessage(content=ai_content))
        return ai_content

    def parse_data(self, ai_content: str) -> Dict[str, Any]:
        """解析AI返回的JSON数据（兼容旧接口，返回 dict）"""
        try:
            return self._parse_raw(ai_content).model_dump()
        except Exception:
            self.parse_error_count += 1
            return self._default_extraction().model_dump()
