"""
ORM 数据模型定义

全部表结构与字段约束集中于此，供建表与各仓储层统一引用。
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone

# ORM 基类：全部模型表继承自此（create_all 按元数据建表）
Base = declarative_base()


def utcnow():
    """统一的时间戳默认值：naive UTC（存储口径，替代已弃用的 datetime.utcnow）

    约定：所有 created_at/updated_at/last_updated 列存 naive UTC 墙钟；
    业务时间列（如排班 start_time/end_time）存北京墙钟。
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

class Technician(Base):
    """理疗师（技师）信息表"""
    __tablename__ = 'technicians'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    gender = Column(String, nullable=True)      # 新增性别字段
    strength = Column(String, nullable=True)    # 新增力气/倾向性字段
    schedules = relationship("TechnicianSchedule", back_populates="technician", cascade="all, delete-orphan")

class TechnicianSchedule(Base):
    """技师排班表：以时段与状态表示技师日程占用"""
    __tablename__ = 'technician_schedules'
    id = Column(Integer, primary_key=True)
    technician_id = Column(Integer, ForeignKey('technicians.id'))
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)  # 'busy' or 'free'
    appointment_id = Column(Integer, nullable=True)
    technician = relationship("Technician", back_populates="schedules")

class KnowledgeDocument(Base):
    """知识库文档表：内容/分类/关键词/向量，is_active 为软删除标记"""
    __tablename__ = 'knowledge_documents'
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    keywords = Column(JSON, nullable=True)  # 存储关键词列表
    embedding = Column(JSON, nullable=True)  # 存储嵌入向量
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    is_active = Column(Integer, default=1)  # 软删除标记

class PdfDocument(Base):
    """PDF 文档注册表：离线入库链路状态机（pending/parsing/chunking/indexing/done/failed）

    doc_id 为 SHA-256(原始文件名) 完整 hex（两级哈希 ID 策略的文档级标识），
    同名重传即文档更新（注册表重置为 pending + Milvus 按 doc_id 清空重入库）。
    """
    __tablename__ = 'pdf_documents'
    doc_id = Column(String, primary_key=True)
    original_filename = Column(String, nullable=False)
    status = Column(String, nullable=False, default='pending')
    chunk_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

class UserBehavior(Base):
    """用户行为流水表：预约/咨询/问询等行为事件（离线记忆整合的回放源）"""
    __tablename__ = 'user_behaviors'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, default='default_user')  # 单用户场景使用默认用户ID
    action_type = Column(String, nullable=False)  # 'appointment', 'consultation', 'inquiry'
    action_data = Column(JSON, nullable=True)  # 存储行为相关的详细数据
    technician_id = Column(Integer, ForeignKey('technicians.id'), nullable=True)
    session_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    technician = relationship("Technician")

class UserPreference(Base):
    """用户偏好表：按类型存偏好值，confidence_score 为置信度（出现次数）"""
    __tablename__ = 'user_preferences'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, default='default_user')
    preference_type = Column(String, nullable=False)  # 'technician', 'time', 'service', 'duration'
    preference_value = Column(String, nullable=False)
    confidence_score = Column(Integer, default=1)  # 偏好的置信度（出现次数）
    last_updated = Column(DateTime, default=utcnow, onupdate=utcnow)

class UserRecommendation(Base):
    """用户推荐表：待发送推荐（技师推荐/回访提醒等），is_sent 标记发送状态"""
    __tablename__ = 'user_recommendations'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, default='default_user')
    recommendation_type = Column(String, nullable=False)  # 'technician_available', 'return_reminder', 'service_suggestion'
    content = Column(Text, nullable=False)
    technician_id = Column(Integer, ForeignKey('technicians.id'), nullable=True)
    is_sent = Column(Integer, default=0)  # 是否已发送
    created_at = Column(DateTime, default=utcnow)
    sent_at = Column(DateTime, nullable=True)
    technician = relationship("Technician")


class ConversationMessage(Base):
    """会话消息持久化表 - 支持多会话隔离与会话恢复"""
    __tablename__ = 'conversation_messages'
    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)


class MemoryEntry(Base):
    """记忆条目表 - 离线整合产出，分层记忆加权召回共用（C4a/C4b）"""
    __tablename__ = 'memory_entries'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, default='default_user', index=True)
    entry_type = Column(String, nullable=False)  # service/time_period/duration/technician/technician_type
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)  # text-embedding-v3 1024 维浮点向量
    importance = Column(Float, default=1.0)
    confidence = Column(Float, default=0.0)
    evidence_count = Column(Integer, default=1)
    distinct_sessions = Column(Integer, default=1)
    sessions = Column(JSON, default=list)  # 贡献证据的 session_id 列表（跨会话次数依据）
    last_evidence_at = Column(DateTime, default=utcnow)
    status = Column(String, default='active')  # 'active' | 'retired'
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class IntegrationWatermark(Base):
    """记忆整合水位检查点表 - 增量回放游标与触发条件判定（C4a）"""
    __tablename__ = 'integration_watermarks'
    user_id = Column(String, primary_key=True)
    last_behavior_id = Column(Integer, default=0)
    last_integration_at = Column(DateTime, nullable=True)
    session_count_at_last_run = Column(Integer, default=0)


class MemoryChangeLog(Base):
    """记忆整合变更日志表 - 幂等审计（C4a）"""
    __tablename__ = 'memory_change_logs'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False, default='default_user', index=True)
    operation = Column(String, nullable=False)  # create/merge/downweight/retire/promote
    entry_id = Column(Integer, nullable=True)  # 关联 MemoryEntry.id；promote 写 UserPreference 时为 0
    detail = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)
