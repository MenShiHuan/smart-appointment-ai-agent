"""
咨询 Agent 离线测试（C1 重写）

原 5 个用例依赖真实 LLM 与真实 embedding 服务（侥幸通过但慢且烧钱）。
重写要点：
- 生成/分类 LLM 经 monkeypatch create_chat_model 注入脚本化 FakeLLM
- 检索器替换为返回固定知识条的假实现；另设一条用例用真实 KnowledgeService
  + FakeEmbedding + 临时 SQLite 验证离线检索链路
- 断言聚焦"知识进入提示词"与"逐 token 流式"等可离线验证的行为
"""
import asyncio

import pytest

from agents.consultant_agent import ConsultantAgent
from services.knowledge_service import KnowledgeService
from tests.conftest import FakeLLM


DOCS = [
    {
        "content": "肩颈理疗可缓解颈椎疼痛与肩膀僵硬，促进局部血液循环。",
        "category": "服务介绍",
        "keywords": ["肩颈理疗", "颈椎", "缓解"],
        "score": 0.92,
        "rank": 1,
    },
    {
        "content": "理疗后请注意保暖，避免受凉。",
        "category": "注意事项",
        "keywords": ["保暖", "受凉"],
        "score": 0.81,
        "rank": 2,
    },
]


class FakeRetriever:
    """返回固定知识条的检索器（隔离真实 embedding / FAISS）"""

    def __init__(self, docs=DOCS):
        self.docs = docs

    async def initialize(self):
        pass

    async def search_knowledge(self, query, top_k=3):
        return self.docs[:top_k]


class NoopBehaviorAgent:
    """no-op 行为记录：流式咨询结束会新建 UserBehaviorAgent 写入真实库，测试中隔离"""

    def record_behavior(self, **kwargs):
        return True


def _make_agent(monkeypatch, llm, docs=DOCS, memory_entries=None):
    """构造咨询 Agent：LLM 与检索器均注入假实现，行为记录 no-op，记忆召回可脚本化"""
    monkeypatch.setattr("agents.consultant_agent.create_chat_model", lambda **kw: llm)
    monkeypatch.setattr("agents.consultant_agent.KnowledgeRetriever", lambda: FakeRetriever(docs))
    # _record_consultation_behavior 内部延迟导入，需 patch 其定义模块
    monkeypatch.setattr("agents.user_behavior_agent.UserBehaviorAgent", lambda: NoopBehaviorAgent())

    class ScriptedMemoryService:
        def __init__(self):
            self.entries = memory_entries or []

        async def recall(self, user_id, query, top_k=None):
            return self.entries

    monkeypatch.setattr("services.memory_service.MemoryService", ScriptedMemoryService)
    return ConsultantAgent()


async def _collect(agen):
    return [t async for t in agen]


def test_consult_answers_with_retrieved_knowledge(monkeypatch):
    """咨询回答：检索知识进入生成提示词，FakeLLM 内容原样返回"""
    fake_llm = FakeLLM(["肩颈理疗每周1-2次有助于缓解疲劳。"])
    agent = _make_agent(monkeypatch, fake_llm)

    response = asyncio.run(agent.consult("肩颈理疗有什么好处？"))

    assert response == "肩颈理疗每周1-2次有助于缓解疲劳。"
    # 检索到的知识应进入生成提示词（RAG 接线验证）
    messages = fake_llm.ainvoke_calls[-1]
    prompt = messages[0]["content"]
    assert "颈椎疼痛与肩膀僵硬" in prompt, "检索知识应注入生成提示词"
    assert "肩颈理疗有什么好处？" in prompt, "用户问题应注入生成提示词"


def test_consult_stream_classifies_then_chunks_answer(monkeypatch):
    """流式咨询：先分类（YES），再逐 token 输出；结束重置状态"""
    from agents.core.session import SessionContext
    from config.constants import StateEnum

    fake_llm = FakeLLM(["YES", "理疗可放松肌肉。"])
    agent = _make_agent(monkeypatch, fake_llm)
    ctx = SessionContext(session_id="s1")

    tokens = asyncio.run(_collect(agent.consult_stream("理疗有什么用？", session=ctx)))

    joined = "".join(tokens)
    assert "理疗可放松肌肉。" in joined
    assert any(t.startswith("[REPLY][咨询机器人]") for t in tokens)
    assert fake_llm.astream_calls, "生成阶段应走流式调用"
    assert ctx.state.value == StateEnum.CLASSIFY, "咨询结束应回分类状态"


def test_consult_stream_unrelated_question_yields_redirect(monkeypatch):
    """无关问题：分类器返回 NO → 输出转交提示，不进入检索生成"""
    fake_llm = FakeLLM(["NO", "不应被调用"])
    agent = _make_agent(monkeypatch, fake_llm)

    tokens = asyncio.run(_collect(agent.consult_stream("今天天气怎么样？")))

    joined = "".join(tokens)
    assert "不是咨询类问题" in joined
    assert not fake_llm.astream_calls, "无关问题不应进入 LLM 生成"


def test_invalid_inputs_do_not_crash(monkeypatch):
    """空/无效输入：不崩溃，产出字符串响应"""
    fake_llm = FakeLLM(["NO"])
    agent = _make_agent(monkeypatch, fake_llm)

    for invalid in ["", "   ", "？", "###"]:
        tokens = asyncio.run(_collect(agent.consult_stream(invalid)))
        response = "".join(tokens)
        assert isinstance(response, str)
        assert len(response) > 0


def test_retriever_searches_knowledge_offline(tmp_path, monkeypatch):
    """检索链路离线验证：真实 KnowledgeService + 关键词感知假向量 + 临时库"""
    class KeywordEmbedding:
        def __call__(self, text):
            return [1.0, 0.0] if "肩颈" in text else [0.0, 1.0]

    monkeypatch.setattr("services.knowledge_service.embed_input", KeywordEmbedding())

    service = KnowledgeService(db_path=f"sqlite:///{tmp_path}/c1_kb.db")
    asyncio.run(service.initialize())

    results = asyncio.run(service.search("肩颈理疗有什么好处？", top_k=3))

    assert isinstance(results, list)
    assert len(results) > 0, "初始化时应播种默认知识并可检索命中"
    for doc in results:
        assert "content" in doc
        assert isinstance(doc["content"], str)
        assert doc.get("score") is not None, "检索结果应带相似度分数"
        assert 0 <= doc["score"] <= 1.0, "归一化后分数应为余弦相似度"


def test_consult_with_empty_knowledge_still_answers(monkeypatch):
    """知识库无命中：仍基于问题本身生成回答（检索结果为空不阻塞）"""
    fake_llm = FakeLLM(["建议您到店咨询理疗师。"])
    agent = _make_agent(monkeypatch, fake_llm, docs=[])

    response = asyncio.run(agent.consult("有什么理疗项目？"))

    assert response == "建议您到店咨询理疗师。"


def test_consult_injects_recalled_memory_context(monkeypatch):
    """C4b：记忆召回命中时，偏好条目注入生成提示词"""
    fake_llm = FakeLLM(["结合您的偏好，建议肩颈理疗。"])
    agent = _make_agent(monkeypatch, fake_llm, memory_entries=[
        {"entry_type": "service", "content": "肩颈理疗"},
        {"entry_type": "time_period", "content": "下午"},
    ])

    response = asyncio.run(agent.consult("最近肩颈不舒服，有什么建议？"))

    assert response == "结合您的偏好，建议肩颈理疗。"
    prompt = fake_llm.ainvoke_calls[-1][0]["content"]
    assert "用户的历史偏好记忆" in prompt
    assert "service=肩颈理疗" in prompt
    assert "time_period=下午" in prompt


def test_consult_without_memory_hits_has_no_memory_section(monkeypatch):
    """C4b：召回为空时不注入记忆段（保持无记忆时的既有提示词结构）"""
    fake_llm = FakeLLM(["建议您到店咨询。"])
    agent = _make_agent(monkeypatch, fake_llm)  # 默认空召回

    asyncio.run(agent.consult("有什么理疗项目？"))

    prompt = fake_llm.ainvoke_calls[-1][0]["content"]
    assert "用户的历史偏好记忆" not in prompt


class FakeTechnicianService:
    """技师表假实现：理疗师档案注入测试（E1 配套）"""

    def get_all_technicians(self):
        return [
            {"name": "李娜", "gender": "女", "strength": "手法细腻，擅长舒缓放松类理疗"},
            {"name": "张伟", "gender": "男", "strength": "擅长肩颈腰背理疗，手法力度大"},
        ]


def test_consult_injects_therapist_profiles_for_therapist_question(monkeypatch):
    """E1 配套：问题涉及理疗师时，技师表档案注入提示词（单一事实源）"""
    fake_llm = FakeLLM(["李娜理疗师擅长舒缓放松类理疗。"])
    monkeypatch.setattr("services.technician_service.TechnicianService", FakeTechnicianService)
    agent = _make_agent(monkeypatch, fake_llm)

    response = asyncio.run(agent.consult("李娜理疗师擅长什么项目？"))

    assert response == "李娜理疗师擅长舒缓放松类理疗。"
    prompt = fake_llm.ainvoke_calls[-1][0]["content"]
    assert "本店理疗师信息" in prompt
    assert "李娜" in prompt and "舒缓放松类理疗" in prompt


def test_consult_skips_therapist_profiles_for_unrelated_question(monkeypatch):
    """E1 配套：与理疗师无关的问题不注入技师表档案（省 Token）"""
    fake_llm = FakeLLM(["营业时间是9点到22点。"])
    monkeypatch.setattr("services.technician_service.TechnicianService", FakeTechnicianService)
    agent = _make_agent(monkeypatch, fake_llm)

    asyncio.run(agent.consult("你们几点开门？"))

    prompt = fake_llm.ainvoke_calls[-1][0]["content"]
    assert "本店理疗师信息" not in prompt


def test_system_prompt_contains_medical_compliance_boundary(monkeypatch):
    """E1 配套：医疗诊断/疗效承诺类问题不做诊断、转医师（合规约束落地）"""
    fake_llm = FakeLLM(["建议您先咨询医师。"])
    agent = _make_agent(monkeypatch, fake_llm)

    asyncio.run(agent.consult("颈椎病做理疗能根治吗？"))

    prompt = fake_llm.ainvoke_calls[-1][0]["content"]
    assert "不做诊断" in prompt
    assert "咨询专业医师" in prompt
