"""
业务服务层模块

包含：知识库（Milvus 双向量混合检索）、向量库、PDF 入库编排（MinerU）、
文本切分、技师、预约、用户行为、记忆整合、推荐调度、文本嵌入等。

包根不做再导出：使用方一律按子模块路径导入
（from services.knowledge_service import get_knowledge_service 等），
避免包根急切导入连带加载 numpy/faiss/schedule 等重依赖。
"""
