"""
配置模块（Shared Kernel）

- constants / domain_terms：常量与领域词表
- memory_config / vector_config：env 可覆盖的运行参数
- model_provider：LLM/Embedding 工厂
- time_config：时间口径与营业时间

包根不做再导出：使用方一律按子模块路径导入
（`from config import memory_config` 亦可用，属子模块回退机制）。
"""
