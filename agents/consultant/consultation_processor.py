"""
咨询流程处理器

负责协调整个咨询流程
"""

from typing import AsyncGenerator, Optional, Dict, Any
from .knowledge_retriever import KnowledgeRetriever
from .consultation_classifier import ConsultationClassifier
from .response_generator import ResponseGenerator

# D1：咨询行为记录的 UserBehaviorAgent 复用单例（懒加载）
_behavior_agent = None


class ConsultationProcessor:
    """咨询流程处理器

    编排标准咨询链路：知识检索（top_k=6）→ 分层记忆召回（C4b）→
    理疗师档案注入（E1）→ 响应生成 → 行为记录（D1）。
    """
    
    def __init__(self, knowledge_retriever: KnowledgeRetriever,
                 consultation_classifier: ConsultationClassifier,
                 response_generator: ResponseGenerator,
                 memory_recaller=None):
        self.knowledge_retriever = knowledge_retriever
        self.consultation_classifier = consultation_classifier
        self.response_generator = response_generator
        # C4b：分层记忆召回器（async callable，返回条目列表；None 表示不注入）
        self.memory_recaller = memory_recaller

    async def _recall_memory_context(self, user_input: str) -> Optional[str]:
        """召回用户偏好记忆（Top-K），失败或为空时不注入"""
        if self.memory_recaller is None:
            return None
        try:
            entries = await self.memory_recaller(user_input)
        except Exception as e:
            print(f"记忆召回失败，跳过注入: {e}")
            return None
        if not entries:
            return None
        return "；".join(f"{e['entry_type']}={e['content']}" for e in entries)

    def _therapist_context(self, user_input: str) -> Optional[str]:
        """理疗师档案注入（E1 配套）：问题涉及理疗师时实时查技师表（单一事实源），
        与知识库双源分离，避免数据不一致。问题点名某理疗师时只注入该理疗师档案
        （减少无关信息干扰，保证模型引用到点名的理疗师）。"""
        related_keywords = ("理疗师", "技师", "老师")
        if not any(kw in user_input for kw in related_keywords):
            return None

        try:
            from services.technician_service import TechnicianService
            technicians = TechnicianService().get_all_technicians()
        except Exception as e:
            print(f"理疗师档案获取失败，跳过注入: {e}")
            return None
        if not technicians:
            return None

        # 点名查询：只注入名字出现在问题里的理疗师
        named = [t for t in technicians
                 if t.get("name") and t["name"] in user_input]
        if named:
            lines = [f"- {t['name']}（{t.get('gender', '')}）：{t.get('strength', '')}"
                     for t in named]
            return "\n".join(lines)

        # 泛指查询（如"店里有女理疗师吗"）：注入全员档案
        lines = [f"- {t['name']}（{t.get('gender', '')}）：{t.get('strength', '')}"
                 for t in technicians if t.get("name")]
        return "\n".join(lines)

    async def process_consultation(self, user_input: str) -> str:
        """处理标准咨询"""
        # 1. 检索知识（top_k=6：多意图问题如"周日开门+地址"需要更多上下文覆盖）
        knowledge_docs = await self.knowledge_retriever.search_knowledge(user_input, top_k=6)

        # 2. 分层记忆召回（C4b）
        memory_context = await self._recall_memory_context(user_input)

        # 3. 理疗师档案注入（E1 配套）
        therapist_context = self._therapist_context(user_input)

        # 4. 生成响应
        response = await self.response_generator.generate_response(
            user_input, knowledge_docs, memory_context=memory_context,
            therapist_context=therapist_context)

        return response

    async def process_consultation_stream(self, user_input: str, session_id: str) -> AsyncGenerator[str, None]:
        """处理流式咨询

        回复逐 token 透出（真流式）；行为记录在流式结束后执行，
        记录失败不影响已输出的回复内容。
        """
        try:
            # 1. 检索知识（top_k=6：多意图问题需要更多上下文覆盖）
            knowledge_docs = await self.knowledge_retriever.search_knowledge(user_input, top_k=6)

            # 2. 分层记忆召回（C4b）
            memory_context = await self._recall_memory_context(user_input)

            # 3. 理疗师档案注入（E1 配套）
            therapist_context = self._therapist_context(user_input)

            # 4. 生成响应
            async for token in self.response_generator.generate_response_stream(
                    user_input, knowledge_docs, memory_context=memory_context,
                    therapist_context=therapist_context):
                yield token
            
            # 3. 记录用户行为
            await self._record_consultation_behavior(user_input, knowledge_docs, session_id)
            
        except Exception as e:
            yield f"[REPLY][咨询机器人]抱歉，处理您的问题时出现了错误：{str(e)}"
    
    async def handle_unrelated_request(self, user_input: str, unrelated_callback, session) -> AsyncGenerator[str, None]:
        """处理与咨询无关的请求（session 为会话上下文）"""
        # 重置状态
        if session is not None:
            from config.constants import StateEnum
            session.state.value = StateEnum.CLASSIFY

        yield self.response_generator.create_unrelated_message()

        # 转给回调处理
        if unrelated_callback:
            async for token in unrelated_callback(user_input, session):
                yield token
    
    async def _record_consultation_behavior(self, user_input: str, knowledge_docs: list, session_id: str):
        """记录咨询行为（D1：行为 Agent 模块级懒加载复用，避免每次咨询重建实例）"""
        try:
            global _behavior_agent
            if _behavior_agent is None:
                from agents.user_behavior_agent import UserBehaviorAgent
                _behavior_agent = UserBehaviorAgent()

            action_data = {
                'question': user_input,
                'knowledge_docs_used': len(knowledge_docs),
                'categories': list(set(doc.get('category', 'unknown') for doc in knowledge_docs)) if knowledge_docs else []
            }

            _behavior_agent.record_behavior(
                action_type='consultation',
                action_data=action_data,
                session_id=session_id
            )
            
        except Exception as behavior_error:
            print(f"记录咨询行为失败：{behavior_error}")
