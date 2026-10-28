"""
Application Module（Web&Application 层组装根）

应用运行时组装：全局 Agent 单例、LangGraph 图/checkpointer 生命周期、
会话恢复与流式管线。API 层经门面（api/chat_handler.py）委托到本包。
"""
