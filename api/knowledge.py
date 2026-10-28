"""
知识库管理API
"""
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
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
    """获取所有知识条目

    GET /api/knowledge/
    响应：documents（已剥离 embedding 向量）、categories、total_count；
    未初始化时自动初始化（建默认数据与向量索引）。
    """
    try:
        # 延迟导入服务单例：首次请求才加载知识库服务（含向量索引等重资源）
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
    """获取特定知识条目

    GET /api/knowledge/{knowledge_id}
    不存在返回 404；响应同样剥离 embedding 向量。
    """
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
    """添加新的知识条目

    POST /api/knowledge/（请求体 KnowledgeItem：content 必填、category/keywords 可选）
    入库时服务层同步生成 embedding 并重建向量索引；data 为 add_document 的布尔结果。
    """
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
    """更新知识条目

    PUT /api/knowledge/{knowledge_id}
    目标不存在返回 404；更新成功后服务层重建向量索引。
    """
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
    """删除知识条目

    DELETE /api/knowledge/{knowledge_id}（服务层软删除：is_active 置 0）
    目标不存在返回 404。
    """
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
    """搜索知识库

    POST /api/knowledge/search：混合检索（BM25 稀疏 + 稠密向量 → RRF 融合 → 语义精排）
    top_k 条相似条目，每条含 score 与 rank（精排分，降级时为归一化 RRF 分）。
    响应：data 为命中条目列表、count 为命中数；
    同时返回 results/query/total_found 字段与 web 搜索页前端契约对齐。
    """
    try:
        from services.knowledge_service import get_knowledge_service
        knowledge_service = get_knowledge_service()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()
        results = await knowledge_service.search(
            request.query, top_k=request.top_k, category=request.category
        )
        results = _strip_embeddings(results)
        return {
            "status": "success",
            "data": results,
            "count": len(results),
            # 前端搜索页实际消费的字段（既有契约错位修复）
            "results": results,
            "query": request.query,
            "total_found": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索知识库失败: {str(e)}")


# ============ PDF 文档离线入库（MinerU → 切分 → Milvus） ============

@router.post("/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """上传 PDF 并提交离线入库任务

    POST /api/knowledge/pdf/upload（multipart/form-data，file 字段）
    立即返回 doc_id（SHA-256 文件名）；解析/切分/入库在后台执行，
    进度经 GET /api/knowledge/pdf/documents/{doc_id}/status 查询。
    同名文件重传即文档更新（先清空旧 chunk 再入库）。
    """
    try:
        from services.knowledge_service import get_knowledge_service
        from services.etl_pipeline import get_etl_pipeline
        # 确保向量库集合就绪（后台任务写入依赖）
        knowledge_service = get_knowledge_service()
        if not knowledge_service.initialized:
            await knowledge_service.initialize()

        filename = file.filename or "document.pdf"
        file_bytes = await file.read()
        doc_id = get_etl_pipeline().submit_pdf(filename, file_bytes)
        return {
            "status": "success",
            "message": "PDF 上传成功，已提交入库任务",
            "data": {"doc_id": doc_id, "original_filename": filename}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 上传失败: {str(e)}")


@router.get("/pdf/documents")
async def list_pdf_documents():
    """获取全部 PDF 文档注册记录（含入库状态与 chunk 数）

    GET /api/knowledge/pdf/documents
    """
    try:
        from services.etl_pipeline import get_etl_pipeline
        documents = get_etl_pipeline().list_documents()
        return {"status": "success", "documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取 PDF 文档列表失败: {str(e)}")


@router.get("/pdf/documents/{doc_id}/status")
async def get_pdf_document_status(doc_id: str):
    """获取单个 PDF 文档的入库状态

    GET /api/knowledge/pdf/documents/{doc_id}/status
    状态机：pending/parsing/chunking/indexing/done/failed
    """
    try:
        from services.etl_pipeline import get_etl_pipeline
        record = get_etl_pipeline().get_status(doc_id)
        if not record:
            raise HTTPException(status_code=404, detail="PDF 文档不存在")
        return {"status": "success", "data": record}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取 PDF 文档状态失败: {str(e)}")


@router.delete("/pdf/documents/{doc_id}")
async def delete_pdf_document(doc_id: str):
    """删除 PDF 文档（清空 Milvus 内全部 chunk + 移除注册记录）

    DELETE /api/knowledge/pdf/documents/{doc_id}
    """
    try:
        from services.etl_pipeline import get_etl_pipeline
        result = await get_etl_pipeline().delete_document(doc_id)
        if not result:
            raise HTTPException(status_code=404, detail="PDF 文档不存在")
        return {"status": "success", "message": "PDF 文档删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除 PDF 文档失败: {str(e)}")
