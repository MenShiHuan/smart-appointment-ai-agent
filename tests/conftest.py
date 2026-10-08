"""
C1 测试离线化共享设施

全部离线测试共用的假组件：
- FakeLLM：按脚本返回文本/Pydantic 对象，支持 ainvoke 与 astream（逐 token）
- FakeEmbedding：确定性向量，替换真实 embedding 调用
- make_fake_session：构造挂载在临时 SQLite 上的会话上下文

原则：任何测试不得发起真实 LLM / embedding / 外部 API 调用。
"""
from typing import List

import pytest

from agents.core.session import SessionContext, SessionRegistry


class _Response:
    """模拟 LLM 响应的最小对象（含 content 属性）"""

    def __init__(self, content):
        self.content = content


class FakeLLM:
    """脚本化假 LLM：ainvoke/astream 共享调用序逐条返回文本；astream 按字符流式输出"""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.ainvoke_calls: List[object] = []
        self.astream_calls: List[object] = []
        self._calls = 0

    def with_structured_output(self, schema, method=None):
        return None  # 模拟 provider 不支持结构化输出（走 prompt+JSON 回退路径）

    async def ainvoke(self, messages):
        self.ainvoke_calls.append(messages)
        idx = min(self._calls, len(self.responses) - 1)
        self._calls += 1
        return _Response(self.responses[idx])

    async def astream(self, messages):
        """逐字符流式输出（模拟真流式的逐 token 产出）"""
        self.astream_calls.append(messages)
        idx = min(self._calls, len(self.responses) - 1)
        self._calls += 1
        for ch in self.responses[idx]:
            yield _Response(ch)


class FakeStructuredLLM:
    """结构化输出模式假 LLM：ainvoke 直接返回 Pydantic 对象（with_structured_output 可用）"""

    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema, method=None):
        return self

    async def ainvoke(self, prompt_text):
        return self.result


class FakeEmbedding:
    """确定性假向量：基于文本哈希生成 8 维向量（无需真实 embedding 服务）"""

    dimension = 8

    def __call__(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        for i, ch in enumerate(text[:64]):
            vec[i % self.dimension] += (ord(ch) % 7) / 7.0
        # 归一化，保持内积语义稳定
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


@pytest.fixture
def fake_llm_factory():
    """构造脚本化 FakeLLM 的工厂（每个测试独立计数）"""
    def _make(responses):
        return FakeLLM(responses)
    return _make


@pytest.fixture
def make_session(tmp_path):
    """构造挂载临时 SQLite 的会话上下文（多会话隔离，不触碰 data/ 真实库）"""
    registry = SessionRegistry(db_path=f"sqlite:///{tmp_path}/c1_sessions.db")

    def _make(session_id="s1") -> SessionContext:
        return registry.get_or_create(session_id)

    _make.registry = registry
    return _make
