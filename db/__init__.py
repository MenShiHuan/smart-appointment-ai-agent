"""
Database Module

数据库模块，包含：
- 数据模型定义（db/models.py）
- 数据访问对象 Repository（db/repositories/）
- 数据库路由器（db/db_router.py）
- 会话管理（db/base/）

包根不做再导出：使用方一律按子模块路径导入
（from db.db_router import DatabaseRouter 等）。
"""
