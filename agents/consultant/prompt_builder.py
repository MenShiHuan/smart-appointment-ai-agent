"""
提示词构建器

负责构建各种类型的提示词
"""

from typing import List, Dict, Any, Optional


class PromptBuilder:
    """提示词构建器"""
    
    def __init__(self):
        self.system_prompt = self._create_system_prompt()
        self.classification_prompt_template = self._create_classification_prompt_template()
    
    def _create_system_prompt(self) -> str:
        """创建系统提示词"""
        return (
            "你是一家健康理疗中心的前台接待员，负责为客户解答关于理疗服务、预约、价格、营业时间、地址、交通等相关问题。"
            "我会为你提供相关的知识库信息，请基于这些信息来回答用户的问题。"
            "如果知识库中没有相关信息，请提供合理的兜底回答，比如："
            "- 对于地址问题：抱歉，具体地址信息请您致电我们门店咨询，我们会详细为您指路。"
            "- 对于交通问题：建议您可以使用地图导航，或者致电我们获取详细的交通指引。"
            "- 对于其他缺失信息：请您直接致电我们或到店咨询，我们会为您提供更详细的信息。"
            "请用专业、礼貌、简洁的语言回复用户。"
            "如果用户的问题与健康理疗服务完全无关（如天气、股票、新闻等），请礼貌地告知用户你只能回答理疗相关问题。"
            "回答时要自然流畅，不要明显地表现出是在查阅资料。"
            "【回答规范】价格类问题必须说明本店服务明码标价、统一收费，可参考价目表菜单并可咨询健康顾问；"
            "会员福利类问题必须同时说明专属折扣优惠与积分兑换服务；"
            "理疗师资质类问题必须说明持证上岗与3年以上专业经验；"
            "用户问题包含多个子问题时，必须逐一回答每个子问题（如同时问营业时间和地址，两个都要给出答案）。"
            "知识库中没有的信息（如医保结算、停车、免费茶水等）一律不得编造肯定或否定的具体结论，"
            "必须礼貌兜底并建议用户致电门店或到店咨询确认。"
            "【合规边界】涉及疾病诊断、治疗效果承诺的问题，一律不做诊断、不做疗效承诺，"
            "建议用户先咨询专业医师；高血压、孕期等特殊人群应提示先经医师评估。"
        )

    def _create_classification_prompt_template(self) -> str:
        """创建分类提示词模板"""
        return (
            "你是一个分类器，判断用户输入是否是关于健康理疗中心的咨询类问题。\n"
            "咨询类问题包括：理疗服务项目、价格、营业时间、理疗师情况、门店地址、交通路线、联系方式、门店环境等。\n"
            "非咨询类问题包括：预约服务（我要预约、帮我安排等）、取消预约、天气、股票、新闻等完全无关的话题。\n"
            "如果是咨询类问题，回答'YES'。如果是预约类问题或完全无关问题，回答'NO'。\n"
            "只回答YES或NO。\n\n"
            "用户输入：{user_input}"
        )
    
    def build_consultation_prompt(self, user_input: str, knowledge_docs: List[Dict[str, Any]],
                                  memory_context: Optional[str] = None,
                                  therapist_context: Optional[str] = None) -> str:
        """构建咨询提示词；memory_context 为分层记忆召回结果（C4b），
        therapist_context 为理疗师档案（咨询链路实时查技师表，E1 配套）"""
        context = self._build_knowledge_context(knowledge_docs)
        memory_section = ""
        if memory_context:
            memory_section = f"\n用户的历史偏好记忆（供参考，如与当前问题无关可忽略）：\n{memory_context}\n"
        therapist_section = ""
        if therapist_context:
            therapist_section = (f"\n本店理疗师信息（回答理疗师相关问题时，必须直接引用"
                                f"以下理疗师信息作答，不得回答\"没有信息\"或建议另行咨询）：\n"
                                f"{therapist_context}\n")
        return (f"{self.system_prompt}\n\n{context}{memory_section}{therapist_section}"
                f"用户问题：{user_input}\n\n请回答用户的问题。")
    
    def build_classification_prompt(self, user_input: str) -> str:
        """构建分类提示词"""
        return self.classification_prompt_template.format(user_input=user_input)
    
    def _build_knowledge_context(self, knowledge_docs: List[Dict[str, Any]]) -> str:
        """构建知识库上下文"""
        if not knowledge_docs:
            return "没有找到直接相关的知识库信息，请基于你对健康理疗服务的专业知识回答。"

        context = "\n以下是相关的知识库信息：\n"
        for i, doc in enumerate(knowledge_docs, 1):
            context += f"{i}. {doc['content']}\n"
        context += ("\n请基于以上信息回答用户问题。如果知识库信息不足以回答问题，"
                    "请按兜底话术引导用户致电门店或到店咨询，不要编造知识库中没有的信息。\n")

        return context
