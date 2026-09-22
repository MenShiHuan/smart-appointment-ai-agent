"""
档期冲突重规划处理器（B3）

预约节点检测到档期冲突（[SIGNAL]replan_needed）后进入 replan 节点：
1. LLM 建议替代时间（失败时确定性顺延 1 小时兜底）
2. 以"模拟用户改约"的输入回流预约节点，走完整的抽取→匹配→落库流程
3. 护栏：最多 2 轮；相同 (理疗师, 时间) 尝试哈希重复即熔断，输出降级话术
"""
import hashlib
import json
from typing import Any, Dict

MAX_REPLAN_ROUNDS = 2

BREAKER_MESSAGE = (
    "[REPLY][预约机器人]很抱歉，您期望的时段附近暂时没有空闲理疗师。"
    "建议您更换其他时间，或直接致电门店为您人工安排。"
)


class ReplanHandler:
    """档期冲突重规划处理器（无状态）"""

    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def attempt_hash(technician_name: str, suggested_time: str) -> str:
        """(理疗师, 建议时间) 尝试的哈希，用于熔断检测"""
        key = f"{technician_name}|{suggested_time}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    async def suggest_time(self, appointment_history: Dict[str, Any]) -> Dict[str, str]:
        """LLM 建议替代时间；失败时确定性顺延 1 小时兜底"""
        from config.time_config import time_config

        prompt = f"""用户想预约理疗服务，但目标时段已满。

用户需求：
- 原预约时间：{appointment_history.get('start_time')}
- 服务时长：{appointment_history.get('duration')}
- 服务项目：{appointment_history.get('project')}

请建议一个邻近的替代时间（当天顺延 1-2 小时，或次日相近时段），
只输出纯JSON，格式：{{"suggested_time": "YYYY-MM-DD HH:MM", "reason": "简短原因"}}"""
        try:
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            raw = response.content if hasattr(response, "content") else str(response)
            # 去除可能的代码块标记
            raw = (raw.strip()
                   .removeprefix("```json")
                   .removeprefix("```")
                   .removesuffix("```")
                   .strip())
            data = json.loads(raw)
            if data.get("suggested_time"):
                return data
            raise ValueError("缺少 suggested_time")
        except Exception as e:
            print(f"[ReplanHandler] LLM 建议失败，使用确定性兜底: {e}")
            try:
                from datetime import timedelta
                dt = time_config.parse_datetime(appointment_history["start_time"])
                return {
                    "suggested_time": (dt + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                    "reason": "顺延一小时",
                }
            except Exception:
                return {}
