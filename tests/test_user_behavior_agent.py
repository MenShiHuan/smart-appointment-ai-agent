"""
用户行为 Agent 离线测试（C1 重写）

原 10 个用例引用的旧组件（recommendation_generator / insight_provider /
behavior_processor 聚合/清理/匿名化）在当前简化版 Agent 中已不存在，
且旧行为记录签名为单 dict 传入。按当前架构重写：
- 组件（BehaviorRecorder/PatternAnalyzer/PreferenceManager）经注入的
  UserBehaviorService 访问临时 SQLite，零真实数据目录触碰
- 记忆查询回答用 FakeLLM，覆盖"有画像"与"无画像"两条路径
"""
import asyncio
from datetime import timedelta

import pytest

from agents.user_behavior.behavior_recorder import BehaviorRecorder
from agents.user_behavior.pattern_analyzer import PatternAnalyzer
from agents.user_behavior.preference_manager import PreferenceManager
from services.user_behavior_service import UserBehaviorService
from tests.conftest import FakeLLM


@pytest.fixture
def service(tmp_path):
    """临时库上的用户行为服务（每个测试独立）"""
    return UserBehaviorService(db_path=f"sqlite:///{tmp_path}/c1_behavior.db")


@pytest.fixture
def recorder(service):
    return BehaviorRecorder(service)


@pytest.fixture
def analyzer(service):
    return PatternAnalyzer(service)


@pytest.fixture
def preferences(service):
    return PreferenceManager(service)


def _record_appointment(service, project="肩颈理疗", technician_id=1):
    """写入一条 appointment 行为记录（created_at 回拨由调用方自行处理）"""
    service.record_behavior(
        user_id="default_user",
        action_type="appointment",
        action_data={"project": project, "duration": "60分钟", "start_time": "2026-09-20 14:00"},
        technician_id=str(technician_id),
        session_id="s-test",
    )


def test_record_and_retrieve_behavior(recorder, service):
    """记录行为 → 可按用户与动作类型检索"""
    result = recorder.record_behavior(
        "appointment_request",
        action_data={"project": "肩颈理疗", "preferred_time": "下午"},
        session_id="s1",
    )

    assert result is not None, "行为记录应返回记录 ID"

    behaviors = service.get_user_behaviors("default_user")
    assert len(behaviors) >= 1
    latest = behaviors[0]
    assert latest["action_type"] == "appointment_request"
    assert latest["action_data"]["project"] == "肩颈理疗"


def test_record_behavior_is_filterable_by_action_type(recorder, service):
    """按动作类型过滤：咨询行为不混入预约行为查询结果"""
    recorder.record_behavior("appointment", action_data={"project": "肩颈理疗"}, session_id="s1")
    recorder.record_behavior("consultation", action_data={"topic": "理疗效果"}, session_id="s1")

    appointments = service.get_user_behaviors("default_user", action_type="appointment")

    assert len(appointments) == 1
    assert all(b["action_type"] == "appointment" for b in appointments)


def test_analyze_preferences_finds_favorite_service(analyzer, service):
    """偏好分析：多数次选择的服务被识别为偏好服务"""
    _record_appointment(service, project="肩颈理疗")
    _record_appointment(service, project="肩颈理疗")
    _record_appointment(service, project="艾灸")

    prefs = analyzer.analyze_user_preferences("default_user")

    assert prefs is not None
    assert prefs["favorite_service"] == "肩颈理疗"
    assert prefs["total_appointments"] == 3
    assert prefs["favorite_technician_id"] is not None


def test_analyze_preferences_new_user_returns_none(analyzer):
    """无历史新用户：偏好分析返回 None，不报错"""
    assert analyzer.analyze_user_preferences("brand_new_user") is None


def test_return_reminder_threshold_logic(analyzer, service, monkeypatch):
    """回访提醒：≥2 次预约且超阈值天数才触发"""
    from datetime import datetime

    _record_appointment(service)
    _record_appointment(service)

    # 新记录（created_at 为当前）→ 不触发
    assert analyzer.should_send_return_reminder("default_user", days_threshold=30) is False

    # 回拨历史记录时间，模拟 40 天未到店 → 触发
    router = service.db_router
    with router.session_manager.session_scope() as session:
        from db.models import UserBehavior as UB
        for row in session.query(UB).filter(UB.user_id == "default_user").all():
            row.created_at = datetime.now() - timedelta(days=40)

    assert analyzer.should_send_return_reminder("default_user", days_threshold=30) is True

    # 不足 2 次预约 → 不触发
    assert analyzer.should_send_return_reminder("brand_new_user", days_threshold=30) is False


def test_generate_return_message_with_favorite_tech(analyzer, service):
    """回访消息：含偏好理疗师姓名与服务项目"""
    class FakeTechService:
        def get_technician_by_id(self, tech_id):
            return {"name": "李医师", "strength": "肩颈理疗"}

    analyzer.technician_service = FakeTechService()
    _record_appointment(service, technician_id=7)

    message = analyzer.generate_return_message("default_user")

    assert message is not None
    assert "李医师" in message
    assert "肩颈理疗" in message


def test_preference_manager_update_and_summary(preferences):
    """偏好存储：更新服务/时间偏好后可从摘要中读回"""
    preferences.update_service_preference("肩颈理疗")
    preferences.update_time_preference("2026-09-20 14:00")

    assert preferences.get_preferred_service() == "肩颈理疗"
    assert preferences.get_preferred_time_period() == "下午"

    summary = preferences.get_preference_summary()
    assert summary["has_service_preference"] is True
    assert summary["preferred_service"] == "肩颈理疗"
    assert summary["preferred_time"] == "下午"


def test_preference_manager_clear(preferences):
    """清除偏好：清除后读不到对应值"""
    preferences.update_service_preference("肩颈理疗")
    preferences.clear_preference("service")

    assert preferences.get_preferred_service() is None


def test_memory_query_answers_from_profile(monkeypatch, tmp_path, service):
    """记忆查询：有画像时经 LLM 基于画像生成回答（FakeLLM 注入）"""
    from agents.user_behavior_agent import UserBehaviorAgent

    fake_llm = FakeLLM(["您常做肩颈理疗，理疗师是李医师。"])
    monkeypatch.setattr("agents.user_behavior_agent.create_chat_model", lambda **kw: fake_llm)
    # UserBehaviorService 在 __init__ 内部延迟导入，需 patch 其定义模块
    monkeypatch.setattr("services.user_behavior_service.UserBehaviorService",
                        lambda: service)

    agent = UserBehaviorAgent()
    _record_appointment(service, technician_id=7)

    tokens = asyncio.run(_collect(agent.answer_preference_query_stream("我常做什么项目？")))

    joined = "".join(tokens)
    assert "您常做肩颈理疗" in joined
    assert fake_llm.astream_calls, "有画像时应调用 LLM 生成回答"


def test_memory_query_new_user_gets_guidance(monkeypatch, tmp_path, service):
    """记忆查询：无画像新用户获得引导语，不调用 LLM"""
    from agents.user_behavior_agent import UserBehaviorAgent

    fake_llm = FakeLLM(["不应被调用"])
    monkeypatch.setattr("agents.user_behavior_agent.create_chat_model", lambda **kw: fake_llm)
    # UserBehaviorService 在 __init__ 内部延迟导入，需 patch 其定义模块
    monkeypatch.setattr("services.user_behavior_service.UserBehaviorService",
                        lambda: service)

    agent = UserBehaviorAgent()

    tokens = asyncio.run(_collect(agent.answer_preference_query_stream("我上次预约的是谁？")))

    joined = "".join(tokens)
    assert "还没有预约记录" in joined
    assert not fake_llm.astream_calls, "无画像时不应调用 LLM"


def test_invalid_behavior_data_degrades_without_crash(recorder):
    """无效行为数据：action_type 为 None 时返回失败标识；
    空字符串被 SQLite NOT NULL 约束接受（仅约束 NULL），写入不崩溃"""
    result = recorder.record_behavior(None, action_data=None, session_id="s-bad")
    assert not result, "action_type=None 应返回失败标识"

    # 空串/空白串：不应抛出异常（DB 层面按字符串接受）
    recorder.record_behavior("", action_data=None, session_id="s-bad")
    recorder.record_behavior("   ", action_data=None, session_id="s-bad")


async def _collect(agen):
    return [t async for t in agen]
