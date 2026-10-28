"""
任务分类器 - 专门负责判断用户请求的类型

职责：
1. 接收用户输入，分析其意图
2. 根据预定义的分类规则，将任务归类为：
   - appointment（预约任务）
   - query（查询任务，服务咨询/价格/理疗师信息等）
   - memory（记忆任务，用户询问自己的历史预约记录或偏好）
   - other（其他任务）
3. 提供清晰的分类结果和置信度
"""

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Dict, Any


class TaskClassifier:
    """任务分类器 - 使用LLM进行智能任务分类"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._initialize_prompt()
        self.chain = self.prompt | self.llm

    def _initialize_prompt(self):
        """初始化分类提示词模板"""
        # 四分类边界提示词：LLM 只允许输出四类英文名，越界由 classify_task 兜底为 other
        self.prompt = PromptTemplate(
            input_variables=["task"],
            template=(
                "你是一个健康理疗预约系统的助手，你的任务是对用户输入的任务进行四分类。\n"
                "类别定义与边界规则：\n"
                "1. appointment（预约任务）：包含预约意图——含\"约/订/预约/安排/订个/报个/有空位/几点能约\"等表述，"
                "或明确想确定预约档期、进行预约、调整预约（改期/提前）。\n"
                "2. query（查询任务）：一切关于本店的业务咨询——服务项目、价格收费、项目效果与适用人群、"
                "理疗师资质与专长、营业时间、地址交通、预约政策、会员优惠、注意事项（做完能否洗澡等）、项目区别对比。\n"
                "3. memory（记忆任务）：用户询问自己的历史记录或个人数据——上次预约的理疗师、常做的项目、"
                "自己的偏好、预约次数、会员积分、健康档案。\n"
                "4. other（其它任务）：与健康理疗业务完全无关的内容（天气、股票、聊天、笑话、外卖等）。\n"
                "只返回类别英文名（appointment/query/memory/other），不要任何其他文字。\n\n"
                "示例：\n"
                "'请帮我预约今天下午3点的肩颈理疗' → appointment\n"
                "'明天下午还有没有空档呀' → appointment\n"
                "'老顾客想约个老地方' → appointment\n"
                "'我想问一下理疗馆在哪里' → query\n"
                "'肩颈理疗对颈椎病有效果吗' → query\n"
                "'办会员卡有什么优惠政策吗' → query\n"
                "'我上次预约的是哪位理疗师' → memory\n"
                "'我的会员积分还有多少' → memory\n"
                "'今天天气怎么样' → other\n"
                "以下是本次归类任务:\n"
                "任务内容：{task}"
            )
        )

    async def classify_task(self, task: str) -> str:
        """
        分类任务

        Args:
            task: 用户输入的任务内容

        Returns:
            str: 分类结果 ('appointment', 'query', 'memory', 'other')
        """
        try:
            category_msg = await self.chain.ainvoke({"task": task})
            # 归一化 LLM 输出：容错大小写与首尾空白
            category = category_msg.content.strip().lower()

            # 验证分类结果是否有效
            valid_categories = {'appointment', 'query', 'memory', 'other'}
            if category not in valid_categories:
                return 'other'  # 默认归类为其他

            return category

        except Exception as e:
            print(f"任务分类失败: {str(e)}")
            return 'other'  # 发生错误时默认归类为其他

    def get_category_description(self, category: str) -> str:
        """获取分类类别的描述信息"""
        descriptions = {
            'appointment': '预约任务 - 用户的预约相关请求',
            'query': '查询任务 - 用户咨询服务项目、价格、理疗师信息等',
            'memory': '记忆任务 - 用户询问自己的历史预约记录或偏好',
            'other': '其他任务 - 与理疗服务无关的请求'
        }
        return descriptions.get(category, '未知任务类型')
