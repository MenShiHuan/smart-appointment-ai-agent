"""
数据库配置模块
"""
import os
from typing import Optional

class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self):
        # 1. 初始化数据库连接路径：优先从环境变量读取 'DATABASE_URL'，若未设置则默认使用本地 SQLite 数据库文件
        self.db_path = os.getenv('DATABASE_URL', 'sqlite:///data/smart_appointment.db')
        # 2. 初始化 SQL 执行日志开关：读取 'DB_ECHO'，转换为布尔值。若为 true，则在控制台打印所有执行的 SQL 语句，方便调试
        self.echo = os.getenv('DB_ECHO', 'false').lower() == 'true'
        # 3. 初始化连接池大小：读取 'DB_POOL_SIZE'，默认大小为 10
        self.pool_size = int(os.getenv('DB_POOL_SIZE', '10'))
        # 4. 初始化最大溢出连接数：读取 'DB_MAX_OVERFLOW'，默认允许超出 pool_size 额外创建 20 个连接
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '20'))
    
    @property
    def connection_string(self) -> str:
        """获取数据库连接字符串"""
        return self.db_path
    
    @property
    def is_sqlite(self) -> bool:
        """判断是否为SQLite数据库"""
        return self.db_path.startswith('sqlite')
    
    def get_engine_kwargs(self) -> dict:
        """获取数据库引擎参数"""
        kwargs = {
            'echo': self.echo,
        }
        
        # SQLite不支持连接池
        if not self.is_sqlite:
            kwargs.update({
                'pool_size': self.pool_size,
                'max_overflow': self.max_overflow,
            })
        
        return kwargs
        
# 创建一个全局唯一的数据库配置实例
db_config = DatabaseConfig()
