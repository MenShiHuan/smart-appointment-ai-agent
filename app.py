"""
FastAPI应用程序

主应用程序入口，配置中间件、路由和异常处理
自动初始化知识库和技师数据
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging

# 导入路由
from api import api_routers
from api.core.exceptions import api_exception_handler, general_exception_handler, BusinessException
from web import router as web_router

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 基于文件位置的绝对路径（不受启动 CWD 影响）
BASE_DIR = Path(__file__).resolve().parent

async def initialize_system():
    """系统启动时自动初始化"""
    try:
        logger.info("🚀 正在初始化智能预约系统...")

        # 初始化知识库服务（与 API 层共享进程级单例，避免重复建索引）
        logger.info("📚 初始化知识库服务...")
        from services.knowledge_service import get_knowledge_service
        knowledge_service = get_knowledge_service()
        await knowledge_service.initialize()

        # 初始化理疗师服务
        logger.info("👨‍⚕️ 初始化理疗师服务...")
        from services.technician_service import TechnicianService
        technician_service = TechnicianService()
        technician_service.initialize_default_technicians()

        # 初始化推荐服务（回访提醒与离线记忆整合调度）
        logger.info("🎯 启动推荐调度服务...")
        from services.recommendation_service import RecommendationService
        recommendation_service = RecommendationService()
        if recommendation_service.start_scheduler():
            logger.info("✅ 推荐调度服务启动成功")
        else:
            logger.warning("⚠️ 推荐调度服务启动失败")

        logger.info("✅ 系统初始化完成！")

    except Exception as e:
        logger.error(f"❌ 系统初始化失败: {e}")
        raise

def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    app = FastAPI(
        title="智能预约AI代理",
        description="提供预约管理、智能咨询、用户行为分析等功能的API服务",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 添加CORS中间件（allow_credentials 与通配来源是非法组合，一期无凭证需求故关闭）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册异常处理器
    from fastapi import HTTPException
    from api.core.exceptions import http_exception_handler
    app.add_exception_handler(BusinessException, api_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # 注册API路由
    for router in api_routers:
        app.include_router(router)

    # 注册Web界面路由
    app.include_router(web_router)

    # 静态文件
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """应用生命周期：启动初始化 + 关闭清理"""
        await initialize_system()
        yield
        # 关闭 checkpointer 连接，避免退出挂起
        from api.chat_handler import close_graph_checkpointer
        await close_graph_checkpointer()

    app.router.lifespan_context = lifespan

    return app

# 创建应用实例
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
