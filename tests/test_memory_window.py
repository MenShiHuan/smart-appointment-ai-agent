"""
分层记忆召回与窗口管理测试（C4b 验收）

全部离线。覆盖 DEV_SPEC 5.4 C4b 验收清单：
1. 加权召回：语义 0.6 + 时效 0.3 + 重要度 0.1 三因子排序生效，Top-K 截断
2. 召回排除 retired 条目
3. 窗口压缩触发阈值（60%）与切分
4. 滚动摘要（含已有摘要滚动累积）
5. 窗口化历史文本（摘要前缀 + 最近 N 条）
6. 预约抽取走窗口化历史（history_text 注入）
7. 预约主流程窗口压缩集成（summary 挂会话上下文、压缩对话进入摘要提示词）
8. 咨询回答注入记忆召回上下文（见 test_consultant_agent.py）
"""
import asyncio
from datetime import datetime, timedelta

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from agents.appointment.input_parser import InputParser
from agents.appointment.schemas import AppointmentExtraction
from agents.appointment_agent import AppointmentAgent
from agents.core.session import (
    MessageWindowManager,
    SessionContext,
    windowed_history_str,
)
from services.memory_service import MemoryService
from tests.conftest import FakeLLM


class DictEmbedding:
    """注册文本 → 正交单位向量；未注册 → 哈希向量（同 test_memory_integration）"""

    def __init__(self, known_texts):
        self.known = {}
        for i, text in enumerate(known_texts):
            vec = [0.0] * len(known_texts)
            vec[i] = 1.0
            self.known[text] = vec
        self.cache = dict(self.known)

    def __call__(self, text):
        if text in self.cache:
            return self.cache[text]
        import hashlib, math
        digest = hashlib.md5(text.encode("utf-8")).digest()
        dim = max(4, len(self.known))
        vec = [0.0] * dim
        for i, byte in enumerate(digest):
            vec[i % dim] += (byte / 255.0 - 0.5) * 2
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        self.cache[text] = vec
        return vec


EMBEDDING_TEXTS = ["肩颈理疗", "足部理疗"]


@pytest.fixture
def recall_service(tmp_path, monkeypatch):
    """MemoryService：临时库 + 确定性 embedding"""
    monkeypatch.setattr("services.memory_service.embed_input",
                        DictEmbedding(EMBEDDING_TEXTS))
    return MemoryService(db_path=f"sqlite:///{tmp_path}/c4b_memory.db", llm=FakeLLM(["noop"]))


def _seed_entry(service, content, entry_type="service", importance=1.0,
                embedding=None, days_old=0, status="active"):
    """直接落一条记忆条目（含可控时效）"""
    entry_id = service.memory_repo.create_entry(
        "default_user", entry_type, content, embedding=embedding,
        session_id="seed")
    service.memory_repo.update_entry(entry_id, importance=importance, status=status)
    if days_old:
        from db.models import utcnow, MemoryEntry
        with service.memory_repo.session_manager.session_scope() as session:
            entry = session.get(MemoryEntry, entry_id)
            entry.last_evidence_at = utcnow() - timedelta(days=days_old)
    return entry_id


# ============ 1. 加权召回排序 ============

def test_recall_scores_by_three_factors(recall_service):
    """加权召回：语义/时效/重要度三因子综合排序（语义命中且新鲜者居首）"""
    emb = DictEmbedding(EMBEDDING_TEXTS)
    # A：语义命中但 10 天前；B：语义无关但新鲜；C：语义命中、新鲜但重要度低
    _seed_entry(recall_service, "肩颈理疗", embedding=emb("肩颈理疗"), days_old=10)
    _seed_entry(recall_service, "足部理疗", embedding=emb("足部理疗"), days_old=0)
    _seed_entry(recall_service, "肩颈理疗", embedding=emb("肩颈理疗"),
                importance=0.25, days_old=0)

    results = asyncio.run(recall_service.recall("default_user", "肩颈理疗"))

    contents = [r["content"] for r in results]
    assert len(results) == 3
    # 语义命中且新鲜的 C 最高（0.6+0.3+0.025=0.925）；A 次之（0.6+0.3×0.9^10+0.1≈0.8046）；B 最低（0.4）
    assert contents[0] == "肩颈理疗"
    assert abs(results[0]["importance"] - 0.25) < 1e-6
    assert abs(results[1]["semantic"] - 1.0) < 1e-6 and results[1]["recency"] < 1.0
    assert results[2]["content"] == "足部理疗"
    assert abs(results[0]["score"] - 0.925) < 1e-6
    assert abs(results[1]["score"] - (0.6 + 0.3 * (0.9 ** 10) + 0.1)) < 1e-6


def test_recall_top_k_and_excludes_retired(recall_service):
    """召回 Top-K 截断，且排除 retired 条目"""
    emb = DictEmbedding(EMBEDDING_TEXTS)
    _seed_entry(recall_service, "肩颈理疗", embedding=emb("肩颈理疗"))
    _seed_entry(recall_service, "足部理疗", embedding=emb("足部理疗"))
    _seed_entry(recall_service, "已退役服务", embedding=emb("肩颈理疗"), status="retired")

    results = asyncio.run(recall_service.recall("default_user", "肩颈理疗", top_k=1))

    assert len(results) == 1
    assert results[0]["content"] == "肩颈理疗"
    assert all(r["content"] != "已退役服务" for r in results)


# ============ 3/4/5. 窗口压缩与滚动摘要 ============

def _messages(count):
    """生成 count 条用户/助手交替消息"""
    out = []
    for i in range(count):
        out.append(HumanMessage(content=f"用户消息{i}"))
        out.append(AIMessage(content=f"助手消息{i}"))
    return out[:count]


def test_window_compress_threshold_and_split():
    """窗口压缩：消息数达 60% 阈值触发，切分为待压缩与保留两段"""
    window = MessageWindowManager(message_limit=20, trigger_ratio=0.6, keep_after_summary=8)

    assert window.should_compress(_messages(11)) is False
    assert window.should_compress(_messages(12)) is True

    to_summarize, kept = window.split_for_compression(_messages(12))
    assert len(to_summarize) == 4, "最早 4 条进入压缩"
    assert len(kept) == 8, "最近 8 条保留"
    assert to_summarize[0].content == "用户消息0"


def test_summarize_rolls_up_existing_summary():
    """滚动摘要：已有摘要并入压缩提示词，输出新摘要"""
    window = MessageWindowManager()
    fake_llm = FakeLLM(["用户想预约肩颈理疗，偏好下午时段。"])

    summary = asyncio.run(window.summarize(
        fake_llm, [HumanMessage(content="我想预约"), AIMessage(content="好的，什么时间？")],
        existing_summary="早前：用户询问过价格"))

    assert summary == "用户想预约肩颈理疗，偏好下午时段。"
    prompt = fake_llm.ainvoke_calls[0][0]["content"]
    assert "已有摘要：早前：用户询问过价格" in prompt, "滚动摘要应把已有摘要并入压缩输入"
    assert "待压缩对话" in prompt


def test_windowed_history_str_keeps_summary_and_recent_only():
    """窗口化历史文本：摘要前缀 + 最近 N 条，更早消息被替代"""
    msgs = _messages(12)
    text = windowed_history_str(msgs, summary="早前聊过理疗项目价格", recent_count=8)

    assert text.startswith("【更早对话摘要】早前聊过理疗项目价格")
    assert "用户消息0" not in text and "助手消息1" not in text, "最早消息应被摘要取代"
    assert "用户消息5" in text and "助手消息5" in text, "保留最近消息"
    assert len([l for l in text.splitlines() if "消息" in l]) == 8


# ============ 6/7. 预约流程窗口集成 ============

def test_input_parser_uses_history_text_override():
    """抽取提示词使用窗口化历史文本，窗口外历史不进入"""
    class CapturingStructuredLLM:
        def with_structured_output(self, schema, method=None):
            return self

        async def ainvoke(self, prompt_text):
            self.last_prompt = prompt_text
            return AppointmentExtraction()

    llm = CapturingStructuredLLM()
    parser = InputParser(llm)
    from langchain_core.chat_history import InMemoryChatMessageHistory
    history = InMemoryChatMessageHistory()
    history.add_message(HumanMessage(content="无关旧消息"))

    asyncio.run(parser.extract("预约", history,
                               history_text="【更早对话摘要】用户咨询过价格\n用户：预约"))

    assert "【更早对话摘要】用户咨询过价格" in llm.last_prompt, "提示词应使用窗口化历史"
    assert "无关旧消息" not in llm.last_prompt, "超出窗口的历史不进入提示词"


def test_run_stream_compresses_window_and_uses_summary(monkeypatch):
    """预约主流程：消息达 60% 触发压缩，摘要挂会话上下文并注入抽取历史"""
    class _Response:
        def __init__(self, content):
            self.content = content

    class DualLLM:
        """窗口摘要与结构化抽取共用的假 LLM：ainvoke 统一返回脚本文本"""

        def __init__(self):
            self.calls = []

        def with_structured_output(self, schema, method=None):
            return self

        async def ainvoke(self, messages):
            self.calls.append(messages)
            return _Response("对话历史摘要内容")

    llm = DualLLM()
    monkeypatch.setattr("agents.appointment_agent.create_chat_model", lambda **kw: llm)

    agent = AppointmentAgent()
    ctx = SessionContext(session_id="s1")
    for i in range(15):
        ctx.chat_history.add_message(HumanMessage(content=f"旧消息{i}"))

    asyncio.run(_collect(agent.run_stream("我想预约理疗", session=ctx)))

    assert ctx.window_summary == "对话历史摘要内容", "摘要应挂到会话上下文"
    prompt_texts = []
    for call in llm.calls:
        if isinstance(call, list):
            prompt_texts.append(call[0]["content"])
        else:
            prompt_texts.append(str(call))
    assert any("待压缩对话" in p for p in prompt_texts), "压缩调用应携带待压缩对话"
    assert any("旧消息0" in p for p in prompt_texts), "最早消息进入摘要输入"
    assert any("【更早对话摘要】对话历史摘要内容" in p for p in prompt_texts), \
        "抽取历史应包含窗口摘要前缀"


async def _collect(agen):
    return [t async for t in agen]
