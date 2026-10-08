"""
知识库管理API
"""
import logging
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["知识库管理"])


def _strip_embeddings(entries):
    """剥离 embedding 向量字段（1024 维向量不应透出给前端）"""
    if not entries:
        return []
    return [{k: v for k, v in doc.items() if k != "embedding"} for doc in entries]


class KnowledgeItem(BaseModel):
    id: Optional[int] = None
    content: str
    category: str = "general"
    keywords: List[str] = []


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    category: Optional[str] = None


@router.get("/")
async def get_all_knowledge():
    """获取所有知识条目"""
    try:
        from services.knowledge_service import get_knowledge_service
        knowledge_service = get_knowledge_service()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        entries = _strip_embeddings(knowledge_service.get_all_documents())

        # 安全获取categories，避免出错
        try:
            categories = knowledge_service.get_all_categories()
        except Exception as e:
            logger.error(f"获取categories失败: {e}")
            categories = []
        
        return {
            "documents": entries or [],
            "categories": categories or [],
            "total_count": len(entries) if entries else 0,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取知识库失败: {str(e)}")


@router.get("/{knowledge_id}")
async def get_knowledge(knowledge_id: int):
    """获取特定知识条目"""
    try:
        from services.knowledge_service import get_knowledge_service
        knowledge_service = get_knowledge_service()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        entry = knowledge_service.get_document(knowledge_id)
        if not entry:
            raise HTTPException(status_code=404, detail="知识条目不存在")
        return {
            "status": "success",
            "data": {k: v for k, v in entry.items() if k != "embedding"}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取知识条目失败: {str(e)}")


@router.post("/")
async def add_knowledge(item: KnowledgeItem):
    """添加新的知识条目"""
    try:
        from services.knowledge_service import get_knowledge_service
        knowledge_service = get_knowledge_service()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        # 直接以内容入库（内容即知识条目全文）
        result = await knowledge_service.add_document(
            content=item.content,
            category=item.category,
            keywords=item.keywords
        )
        return {
            "status": "success",
            "message": "知识条目添加成功",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加知识条目失败: {str(e)}")


@router.put("/{knowledge_id}")
async def update_knowledge(knowledge_id: int, item: KnowledgeItem):
    """更新知识条目"""
    try:
        from services.knowledge_service import get_knowledge_service
        knowledge_service = get_knowledge_service()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        result = await knowledge_service.update_document(
            doc_id=knowledge_id,
            content=item.content,
            category=item.category,
            keywords=item.keywords
        )
        if not result:
            raise HTTPException(status_code=404, detail="知识条目不存在")
        return {
            "status": "success",
            "message": "知识条目更新成功",
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新知识条目失败: {str(e)}")


@router.delete("/{knowledge_id}")
async def delete_knowledge(knowledge_id: int):
    """删除知识条目"""
    try:
        from services.knowledge_service import get_knowledge_service
        knowledge_service = get_knowledge_service()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        result = await knowledge_service.delete_document(knowledge_id)
        if not result:
            raise HTTPException(status_code=404, detail="知识条目不存在")
        return {
            "status": "success",
            "message": "知识条目删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除知识条目失败: {str(e)}")


@router.post("/search")
async def search_knowledge(request: SearchRequest):
    """搜索知识库"""
    try:
        from services.knowledge_service import get_knowledge_service
        knowledge_service = get_knowledge_service()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        results = await knowledge_service.search(request.query, top_k=request.top_k)
        return {
            "status": "success",
            "data": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索知识库失败: {str(e)}")
