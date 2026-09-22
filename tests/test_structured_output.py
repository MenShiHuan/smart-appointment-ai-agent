"""
结构化输出与修复重试测试（A4 验收）

全部离线：FakeLLM 按脚本返回原始文本/Pydantic 对象，不调真实 API。
覆盖：
1. 首次输出非法 JSON → 修复重试成功
2. 重试仍失败 → 降级默认值 + parse_error 计数
3. provider 支持结构化输出时走结构化路径
4. 会话历史记录 user/assistant 消息
5. parse_data 兼容接口降级行为
"""
import asyncio
import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory

from agents.appointment.input_parser import InputParser
from agents.appointment.schemas import AppointmentExtraction


class _Response:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """prompt+JSON 模式的假 LLM：按脚本返回原始文本"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def with_structured_output(self, schema):
        return None  # 模拟 provider 不支持结构化输出

    async def ainvoke(self, messages):
        idx = min(self.call_count, len(self.responses) - 1)
        content = self.responses[idx]
        self.call_count += 1
        return _Response(content)


class FakeStructuredLLM:
    """结构化输出模式的假 LLM：ainvoke 直接返回 Pydantic 对象"""

    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, prompt_text):
        return self.result


VALID_JSON = ('{"gender": "女", "start_time": "2026-09-23 15:00", "duration": "60分钟", '
              '"project": "肩颈理疗", "preference": "无", "technician_name": "未知", '
              '"confirmation": "未知", "info_complete": false, "unrelated": false, "missing_info": []}')


def make_history():
    return InMemoryChatMessageHistory()


def test_repair_retry_recovers_from_invalid_json():
    """首次输出非法 JSON → 修复重试成功"""
    llm = FakeLLM(["```json 这不是合法JSON", VALID_JSON])
    parser = InputParser(llm)

    extraction = asyncio.run(parser.extract("明天下午3点做肩颈理疗", make_history()))

    assert extraction.project == "肩颈理疗"
    assert extraction.gender == "女"
    assert extraction.start_time == "2026-09-23 15:00"
    assert llm.call_count == 2  # 首次失败 + 修复重试
    assert parser.parse_error_count == 0


def test_retry_exhausted_degrades_to_defaults():
    """重试仍失败 → 降级默认值 + parse_error 计数"""
    llm = FakeLLM(["无效输出1", "无效输出2"])
    parser = InputParser(llm)

    extraction = asyncio.run(parser.extract("你好", make_history()))

    assert extraction.info_complete is False
    assert extraction.unrelated is False
    assert extraction.start_time == "未知"
    assert extraction.missing_info == ["所有信息"]
    assert parser.parse_error_count == 1
    assert llm.call_count == 2


def test_structured_output_path_used_when_supported():
    """provider 支持结构化输出时直接走结构化路径"""
    expected = AppointmentExtraction(
        project="艾灸", start_time="2026-09-23 10:00", info_complete=True
    )
    parser = InputParser(FakeStructuredLLM(expected))

    extraction = asyncio.run(parser.extract("预约艾灸", make_history()))

    assert extraction == expected
    assert parser.parse_error_count == 0


def test_history_records_user_and_assistant_messages():
    """会话历史正确记录 user/assistant 消息"""
    llm = FakeLLM([VALID_JSON])
    parser = InputParser(llm)
    history = make_history()

    asyncio.run(parser.extract("明天下午3点做肩颈理疗", history))

    types = [m.type for m in history.messages]
    assert types == ["human", "ai"]


def test_parse_data_compat_degrades_with_counting():
    """兼容接口 parse_data 失败时降级并计数"""
    llm = FakeLLM([VALID_JSON])
    parser = InputParser(llm)

    data = parser.parse_data("不是JSON")

    assert data["gender"] == "未知"
    assert data["info_complete"] is False
    assert parser.parse_error_count == 1
