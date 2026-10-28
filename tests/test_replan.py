"""
档期冲突自愈循环测试（B3 验收）

全部离线。场景：
1. 首轮重规划成功：冲突 → replan 建议替代时间 → 回流预约节点匹配成功
2. 相同建议重复 → 哈希熔断，输出降级话术
3. 建议不同但持续冲突 → 轮数上限（≤2）熔断
单元测试：attempt_hash、suggest_time 解析与确定性兜底。
"""
import asyncio
import pytest

from agents.core.session import SessionRegistry
from agents.graph import build_agent_graph, stream_graph_task
from agents.graph.replan import ReplanHandler, BREAKER_MESSAGE


class _R:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """按脚本返回原始文本的假 LLM"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    async def ainvoke(self, messages):
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return _R(self.responses[idx])


SUGGEST_1600 = '{"suggested_time": "2026-09-23 16:00", "reason": "顺延一小时"}'
SUGGEST_1700 = '{"suggested_time": "2026-09-23 17:00", "reason": "再顺延一小时"}'


class ConflictAppointmentAgent:
    """预约 Agent stub：前 N 次输出冲突信号，之后输出成功"""

    def __init__(self, llm, conflict_times=None):
        self.llm = llm
        self.conflict_times = conflict_times if conflict_times is not None else 10**9
        self.calls = 0
        self.last_user_input = None

    async def run_stream(self, user_input=None, session=None,
                         enable_replan_signal=False, enable_confirm_signal=False):
        self.calls += 1
        self.last_user_input = user_input
        if self.calls <= self.conflict_times:
            yield "[SIGNAL]replan_needed"
            yield "[REPLY][预约机器人]抱歉该时段没空"  # 应被节点抑制
        else:
            yield f"[REPLY][预约机器人]已为您预约成功（改约输入：{user_input}）"


class StubConsultant:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def consult_stream(self, user_input, session=None):
        yield "[REPLY][咨询机器人]stub"


class StubTaskAgent:
    def __init__(self, appointment_agent, db_url):
        self.task_classifier = type("C", (), {"classify_task": self._classify})()
        self.appointment_agent = appointment_agent
        self.consultant_agent = StubConsultant()
        self.behavior_agent = None
        self.session_registry = SessionRegistry(db_path=db_url)

    async def _classify(self, task):
        return "appointment"


async def _run(tmp_path, task_agent, sid):
    """以固定输入运行一轮图任务，返回拼接后的全部 token"""
    graph = build_agent_graph(task_agent)
    tokens = []
    async for token in stream_graph_task(graph, "我想预约今天下午3点的理疗", sid):
        tokens.append(token)
    return "".join(tokens)


# ============ 场景测试 ============

def test_first_replan_succeeds(tmp_path):
    """冲突 → 建议替代时间 → 回流匹配成功；失败消息被抑制"""
    llm = FakeLLM([SUGGEST_1600])
    agent = ConflictAppointmentAgent(llm, conflict_times=1)
    task_agent = StubTaskAgent(agent, f"sqlite:///{tmp_path}/g.db")

    joined = asyncio.run(_run(tmp_path, task_agent, "s1"))

    # 冲突失败消息被抑制，替代为自愈流程
    assert "抱歉该时段没空" not in joined
    assert "尝试为您调整到 2026-09-23 16:00" in joined
    assert "已为您预约成功" in joined
    # 回流预约节点的输入携带新时间
    assert "改约到 2026-09-23 16:00" in agent.last_user_input
    assert agent.calls == 2
    assert llm.call_count == 1


def test_repeated_suggestion_circuit_breaks(tmp_path):
    """LLM 连续建议同一时间 → 哈希熔断"""
    llm = FakeLLM([SUGGEST_1600, SUGGEST_1600])
    agent = ConflictAppointmentAgent(llm, conflict_times=10)
    task_agent = StubTaskAgent(agent, f"sqlite:///{tmp_path}/g.db")

    joined = asyncio.run(_run(tmp_path, task_agent, "s1"))

    assert BREAKER_MESSAGE in joined
    assert "致电门店" in joined
    # 第一次建议正常透出，第二次（重复）熔断
    assert joined.count("尝试为您调整到 2026-09-23 16:00") == 1
    assert llm.call_count == 2


def test_loop_cap_circuit_breaks_after_two_rounds(tmp_path):
    """建议不同但持续冲突 → 轮数上限（≤2）熔断"""
    llm = FakeLLM([SUGGEST_1600, SUGGEST_1700])
    agent = ConflictAppointmentAgent(llm, conflict_times=10)
    task_agent = StubTaskAgent(agent, f"sqlite:///{tmp_path}/g.db")

    joined = asyncio.run(_run(tmp_path, task_agent, "s1"))

    assert BREAKER_MESSAGE in joined
    assert "尝试为您调整到 2026-09-23 16:00" in joined
    assert "尝试为您调整到 2026-09-23 17:00" in joined
    assert llm.call_count == 2          # 建议恰好 2 轮
    assert agent.calls == 3             # 初始 + 2 次回流重试


def test_no_conflict_keeps_normal_flow(tmp_path):
    """无冲突时不进入 replan，也不产生循环计数"""
    llm = FakeLLM([SUGGEST_1600])
    agent = ConflictAppointmentAgent(llm, conflict_times=0)
    task_agent = StubTaskAgent(agent, f"sqlite:///{tmp_path}/g.db")

    joined = asyncio.run(_run(tmp_path, task_agent, "s1"))

    assert "已为您预约成功" in joined
    assert "尝试为您调整" not in joined
    assert llm.call_count == 0          # 未触发 replan，未调用建议 LLM


# ============ 单元测试 ============

def test_attempt_hash_deterministic_and_distinct():
    """尝试哈希：同技师同时段哈希一致，不同时段哈希不同（幂等熔断依据）"""
    h1 = ReplanHandler.attempt_hash("张伟", "2026-09-23 16:00")
    h2 = ReplanHandler.attempt_hash("张伟", "2026-09-23 16:00")
    h3 = ReplanHandler.attempt_hash("张伟", "2026-09-23 17:00")
    assert h1 == h2
    assert h1 != h3


def test_suggest_time_parses_json_with_code_fences():
    """建议解析：代码围栏包裹的 JSON 建议可正确解析"""
    llm = FakeLLM(['```json\n{"suggested_time": "2026-09-23 16:00", "reason": "顺延"}\n```'])
    handler = ReplanHandler(llm)
    result = asyncio.run(handler.suggest_time({"start_time": "2026-09-23 15:00"}))
    assert result["suggested_time"] == "2026-09-23 16:00"


def test_suggest_time_falls_back_to_plus_one_hour():
    """LLM 输出不可解析 → 确定性顺延 1 小时"""
    llm = FakeLLM(["这不是JSON"])
    handler = ReplanHandler(llm)
    result = asyncio.run(handler.suggest_time({
        "start_time": "2026-09-23 15:00",
        "duration": "60分钟",
        "project": "肩颈理疗",
    }))
    assert result["suggested_time"] == "2026-09-23 16:00"
    assert "顺延" in result["reason"]


def test_suggest_time_returns_empty_when_time_unparseable():
    """原时段不可解析：返回空建议"""
    llm = FakeLLM(["garbage"])
    handler = ReplanHandler(llm)
    result = asyncio.run(handler.suggest_time({"start_time": "未知"}))
    assert result == {}
