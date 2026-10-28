"""
会话状态定义

StateEnum：会话状态机（图节点路由依据）；SharedState：会话级状态容器。
"""
from enum import Enum

class StateEnum(Enum):
    """会话状态机枚举：CLASSIFY 为默认分类态，其余为对应业务流状态"""
    CLASSIFY = "classify"
    APPOINTMENT = "appointment"
    CONSULT = "consult"
    MEMORY = "memory"
    
class SharedState:
    """会话状态容器：value 保存当前状态，新会话默认从 CLASSIFY 开始"""
    def __init__(self):
        self.value = StateEnum.CLASSIFY
