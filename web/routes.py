"""
Web界面路由

处理前端页面渲染和聊天功能
"""
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from api.chat_handler import ProcessUserInput_stream
import logging

# 创建logger实例
logger = logging.getLogger(__name__)
# 模板配置（绝对路径，不受启动 CWD 影响）
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

# Web路由器
router = APIRouter(tags=["Web界面"])

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

@router.get("/", response_class=HTMLResponse, summary="主页")
async def read_root(request: Request):
    """渲染主页聊天界面"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/metrics", summary="进程内指标（F1）")
async def metrics():
    """轻量指标端点：请求计数与阶段耗时统计"""
    from services.trace import metrics_registry
    return metrics_registry.snapshot()

@router.post("/chat/stream", summary="流式聊天")
async def chat_stream_endpoint(chat: ChatRequest):
    """处理流式聊天请求（按 session_id 隔离会话）"""
    if chat.session_id and not _SESSION_ID_PATTERN.match(chat.session_id):
        raise HTTPException(status_code=400, detail="session_id 格式非法")

    async def token_generator():
        try:
            async for token in ProcessUserInput_stream(chat.message, session_id=chat.session_id):
                yield token
        except Exception:
            # 响应已开始后无法改状态码：以错误标记收尾，避免客户端静默截断
            logger.exception("流式聊天处理异常")
            yield "[ERROR]服务处理您的请求时发生异常，请稍后重试。"
    return StreamingResponse(token_generator(), media_type="text/plain")

@router.get("/user_behavior", response_class=HTMLResponse, summary="用户行为分析页面")
async def user_behavior_page(request: Request):
    """用户行为分析页面"""
    return templates.TemplateResponse("user_behavior_analysis.html", {"request": request})

@router.get("/knowledge", response_class=HTMLResponse, summary="知识库管理页面")
async def knowledge_page(request: Request):
    """知识库管理页面"""
    # 通过API层获取知识库数据
    try:
        from api.knowledge import get_all_knowledge
        
        # 调用API层函数获取数据
        knowledge_data = await get_all_knowledge()
        documents = knowledge_data.get("documents", [])
        categories = knowledge_data.get("categories", [])
        
        return templates.TemplateResponse("knowledge_management.html", {
            "request": request,
            "documents": documents,
            "categories": categories
        })
    except Exception as e:
        return templates.TemplateResponse("knowledge_management.html", {
            "request": request,
            "documents": [],
            "categories": [],
            "error": str(e)
        })

@router.get("/technician", response_class=HTMLResponse, summary="理疗师状态页面")
async def technician_page(request: Request):
    """理疗师状态页面"""
    # 通过API层获取理疗师数据
    try:
        from api.technician import get_all_technicians
        
        # 调用API层函数获取数据
        technicians = await get_all_technicians()
        
        return templates.TemplateResponse("technician.html", {
            "request": request,
            "technicians": technicians
        })
    except Exception as e:
        return templates.TemplateResponse("technician.html", {
            "request": request,
            "technicians": [],
            "error": str(e)
        })

@router.get("/technician_schedule", response_class=HTMLResponse, summary="理疗师排班页面")
async def technician_schedule_page(request: Request):
    """理疗师排班页面"""
    try:
        from api.technician import get_all_technicians_schedule_today
        from config.time_config import time_config
        
        # 获取当前日期
        current_date = time_config.current_date_str()
        
        # 通过API层获取所有理疗师的排班数据
        schedules_data = await get_all_technicians_schedule_today()
        
        # 构建排班数据格式 - 直接使用API返回的数据
        schedule = []
        for schedule_item in schedules_data:
            schedule.append({
                "id": schedule_item["technician_id"],
                "name": schedule_item["technician_name"],
                "busy_periods": schedule_item["busy_periods"]
            })
        
        return templates.TemplateResponse("technician_schedule.html", {
            "request": request,
            "schedule": schedule,
            "current_date": current_date
        })
    except Exception as e:
        logger.error(f"加载理疗师排班数据失败: {str(e)}")
        return templates.TemplateResponse("technician_schedule.html", {
            "request": request,
            "schedule": [],
            "error": str(e)
        })

@router.get("/user_behavior_analysis", response_class=HTMLResponse, summary="用户行为分析页面")
async def user_behavior_analysis_page(request: Request):
    """用户行为分析页面"""
    return templates.TemplateResponse("user_behavior_analysis.html", {"request": request})
