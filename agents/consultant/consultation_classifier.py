"""
咨询分类器

负责判断用户输入是否为咨询类问题
"""

from langchain_core.language_models.chat_models import BaseChatModel
from .prompt_builder import PromptBuilder


class ConsultationClassifier:
    """咨询分类器"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt_builder = PromptBuilder()
    
    async def is_consultation_related(self, user_input: str) -> bool:
        """检查用户输入是否与咨询相关"""
        try:
            prompt = self.prompt_builder.build_classification_prompt(user_input)
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            # 归一化判定：容忍标点与"是"等变体，避免 LLM 输出微小差异导致误判
            result = response.content.strip().strip("。，！？!?., ")
            return result.upper().startswith("YES") or result.startswith("是")
        except Exception as e:
            print(f"分类判断出错：{e}")
            # 如果分类出错，默认认为是相关的，避免误判
            return True
