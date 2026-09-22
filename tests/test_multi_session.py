"""
多会话隔离测试（A1 验收）

验证核心目标：
1. 不同 session_id 的会话状态互不污染
2. 会话消息持久化与恢复（模拟服务重启）
3. 清空会话同步清理数据库
4. 会话级轮换索引隔离

全部离线运行，不依赖 LLM API。
"""
import pytest
from langchain_core.messages import HumanMessage

from agents.core.session import SessionRegistry
from config.constants import StateEnum


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path}/test_sessions.db"


@pytest.fixture
def registry(db_url):
    return SessionRegistry(db_path=db_url)


def test_distinct_sessions_do_not_share_state(registry):
    """两个并发会话的状态互不污染"""
    ctx1 = registry.get_or_create("session-a")
    ctx2 = registry.get_or_create("session-b")

    assert ctx1 is not ctx2
    assert ctx1.state is not ctx2.state
    assert ctx1.chat_history is not ctx2.chat_history
    assert ctx1.appointment_history is not ctx2.appointment_history

    # 修改会话A的状态，会话B不受影响
    ctx1.state.value = StateEnum.APPOINTMENT
    ctx1.appointment_history["project"] = "肩颈理疗"

    assert ctx2.state.value == StateEnum.CLASSIFY
    assert ctx2.appointment_history["project"] is None


def test_same_session_id_returns_same_context(registry):
    ctx1 = registry.get_or_create("session-a")
    ctx2 = registry.get_or_create("session-a")
    assert ctx1 is ctx2


def test_chat_messages_persist_and_restore(registry, db_url):
    """服务重启后会话历史可恢复"""
    ctx = registry.get_or_create("session-a")
    ctx.chat_history.add_message(HumanMessage(content="你好"))

    # 模拟服务重启：新建注册表指向同一个数据库
    registry2 = SessionRegistry(db_path=db_url)
    restored = registry2.get_or_create("session-a")
    contents = [m.content for m in restored.chat_history.messages]
    assert contents == ["你好"]


def test_clear_history_deletes_persisted_messages(registry, db_url):
    """清空会话同时清理数据库消息"""
    ctx = registry.get_or_create("session-a")
    ctx.chat_history.add_message(HumanMessage(content="你好"))
    ctx.chat_history.clear()

    registry2 = SessionRegistry(db_path=db_url)
    restored = registry2.get_or_create("session-a")
    assert len(restored.chat_history.messages) == 0


def test_restored_context_does_not_duplicate_messages(registry, db_url):
    """已加载的消息不会被二次持久化"""
    ctx = registry.get_or_create("session-a")
    ctx.chat_history.add_message(HumanMessage(content="你好"))

    registry2 = SessionRegistry(db_path=db_url)
    ctx2 = registry2.get_or_create("session-a")
    ctx2.chat_history.add_message(HumanMessage(content="在吗"))

    registry3 = SessionRegistry(db_path=db_url)
    restored = registry3.get_or_create("session-a")
    contents = [m.content for m in restored.chat_history.messages]
    assert contents == ["你好", "在吗"]


def test_unrelated_reply_rotation_is_per_session(registry):
    ctx1 = registry.get_or_create("session-a")
    ctx2 = registry.get_or_create("session-b")
    ctx1.unrelated_reply_index = 5
    assert ctx2.unrelated_reply_index == 0


def test_new_session_id_is_unique(registry):
    ids = {registry.new_session_id() for _ in range(100)}
    assert len(ids) == 100
