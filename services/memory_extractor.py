"""
离线记忆整合 - 偏好候选抽取器（C4a）

（自 agents/user_behavior/ 移入 Services 层：纯抽取工具，唯一消费者是
services/memory_service.py，移入后消除 Services→Agents 的唯一模块级反向边。）

混合抽取策略：
1. appointment 行为 → 规则抽取（action_data 已结构化，零 LLM 成本）
2. consultation 行为 → LLM 结构化抽取（question 自由文本中的偏好信号），
   失败丢弃该条、不阻塞整合流程

候选字段与 MemoryEntry 的 entry_type 对齐：
service / time_period / duration / technician / technician_type
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from services.json_utils import extract_json


@dataclass
class PreferenceCandidate:
    """偏好候选：整合流程中的中间产物"""
    entry_type: str
    content: str
    behavior_id: int
    session_id: Optional[str] = None


# ============ 规则抽取（appointment 行为） ============

def _time_period_from_start_time(start_time: str) -> Optional[str]:
    """按开始时间映射时段：6-11 上午 / 12-17 下午 / 其余 晚上"""
    if not start_time:
        return None
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    # 时段边界与 preference_manager.update_time_preference 的划分保持一致
    if 6 <= dt.hour < 12:
        return "上午"
    if 12 <= dt.hour < 18:
        return "下午"
    return "晚上"


def extract_from_behavior(behavior: Dict[str, Any]) -> List[PreferenceCandidate]:
    """从 appointment 行为规则抽取偏好候选（consultation 走 LLM，此处返回空）"""
    action_type = behavior.get("action_type")
    if action_type != "appointment":
        return []

    data = behavior.get("action_data") or {}
    candidates: List[PreferenceCandidate] = []

    if data.get("project"):
        candidates.append(PreferenceCandidate(
            "service", data["project"], behavior["id"], behavior.get("session_id")))
    if data.get("start_time"):
        period = _time_period_from_start_time(data["start_time"])
        if period:
            candidates.append(PreferenceCandidate(
                "time_period", period, behavior["id"], behavior.get("session_id")))
    if data.get("duration"):
        candidates.append(PreferenceCandidate(
            "duration", str(data["duration"]), behavior["id"], behavior.get("session_id")))
    if behavior.get("technician_id"):
        candidates.append(PreferenceCandidate(
            "technician", str(behavior["technician_id"]), behavior["id"],
            behavior.get("session_id")))
    if data.get("preference"):
        candidates.append(PreferenceCandidate(
            "technician_type", data["preference"], behavior["id"],
            behavior.get("session_id")))

    return candidates


# ============ LLM 结构化抽取（consultation 行为） ============

class ExtractedPreference(BaseModel):
    """单条偏好（LLM 结构化输出约束）"""
    preference_type: str = Field(description="service/time_period/duration/technician/technician_type")
    content: str = Field(description="偏好内容，如 肩颈理疗/下午/60分钟")


class PreferenceExtractionResult(BaseModel):
    """偏好抽取结果：无偏好信号时列表为空"""
    preferences: List[ExtractedPreference] = Field(default_factory=list)


EXTRACTION_PROMPT = """你是一个用户偏好信号抽取器。从用户的咨询文本中提取与理疗服务相关的偏好信号。

可提取的偏好类型（无信号则对应项不输出）：
- service：用户关注或偏好的服务项目（如 肩颈理疗、艾灸）
- time_period：用户偏好的时间段（上午/下午/晚上）
- duration：用户偏好的服务时长（如 60分钟、90分钟）
- technician：用户提到的理疗师（姓名或编号）
- technician_type：用户的手法偏好（如 手法轻、力度大）

要求：
1. 只输出纯JSON，不要任何代码块标记或其他文字
2. 没有偏好信号时输出 {{"preferences": []}}
3. content 只保留偏好本身，去除无关描述

用户咨询文本：
{question}"""


class ConsultationExtractor:
    """consultation 行为的 LLM 偏好抽取（结构化输出 + JSON 修复重试）"""

    def __init__(self, llm):
        self.llm = llm
        self._structured_llm = None
        try:
            # DeepSeek 仅支持 JSON Output 模式（见 input_parser 说明）
            self._structured_llm = llm.with_structured_output(
                PreferenceExtractionResult, method="json_mode")
        except Exception:
            self._structured_llm = None

    async def extract(self, question: str, behavior: Dict[str, Any]) -> List[PreferenceCandidate]:
        """从咨询文本抽取偏好候选；抽取失败返回空列表（丢弃该条，不阻塞整合流程）"""
        result = await self._try_extract(question)
        if result is None:
            return []
        # 白名单过滤：只接受与 MemoryEntry.entry_type 对齐的五类，防御 LLM 越界输出
        return [
            PreferenceCandidate(p.preference_type, p.content, behavior["id"],
                                behavior.get("session_id"))
            for p in result.preferences
            if p.content and p.preference_type
            in {"service", "time_period", "duration", "technician", "technician_type"}
        ]

    async def _try_extract(self, question: str) -> Optional[PreferenceExtractionResult]:
        """结构化输出优先，失败回退 prompt+JSON 修复重试；全部失败返回 None"""
        prompt = EXTRACTION_PROMPT.format(question=question)

        # 1. 结构化输出路径
        if self._structured_llm is not None:
            try:
                result = await self._structured_llm.ainvoke(prompt)
                if isinstance(result, PreferenceExtractionResult):
                    return result
            except Exception as e:
                print(f"[MemoryExtractor] 结构化抽取失败，进入 JSON 回退: {e}")

        # 2. prompt+JSON 路径（含一次修复重试；统一提取工具，见 services/json_utils.py）
        try:
            raw = await self._invoke_raw(prompt)
            try:
                data = extract_json(raw)
                return PreferenceExtractionResult.model_validate(data)
            except Exception as first_error:
                # 回显截断：只回带输出片段，避免异常长输出撑大修复提示
                repair_prompt = (prompt + f"\n你上次的输出无法解析：{raw[:200]}"
                                 f"\n错误信息：{first_error}\n请直接输出纯JSON。")
                raw_retry = await self._invoke_raw(repair_prompt)
                try:
                    data = extract_json(raw_retry)
                    return PreferenceExtractionResult.model_validate(data)
                except Exception:
                    return None
        except Exception:
            # LLM 不可用：丢弃该条候选，不阻塞整合
            return None

    async def _invoke_raw(self, prompt_text: str) -> str:
        """以用户消息调用 LLM 并返回原始文本输出"""
        response = await self.llm.ainvoke([{"role": "user", "content": prompt_text}])
        return response.content if hasattr(response, "content") else str(response)
