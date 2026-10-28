"""
PDF 离线入库编排（链路 1）：上传 → MinerU 解析 → 结构感知切分 → Milvus 入库

- SQLite 注册表状态机：pending → parsing → chunking → indexing → done / failed
- submit_pdf 立即返回 doc_id，解析在后台任务执行，进度经状态表查询
- doc_id = SHA-256(原始文件名) 完整 hex（两级哈希 ID 策略的文档级标识）：
  同名重传即文档更新（先按 doc_id 清空旧 chunk 再入库，chunk_id 确定性幂等）
"""
import asyncio
import hashlib
import logging
import threading
from typing import List, Dict, Optional

from db.db_router import DatabaseRouter
from services.mineru_service import MinerUService
from services.text_chunker import chunk_document
from services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


# ============ 进程级单例（API 层与后台任务共享同一编排器） ============
_instance = None
_instance_lock = threading.Lock()


def get_etl_pipeline() -> "EtlPipeline":
    """获取进程级 PDF 入库编排器单例（懒加载，线程安全）"""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = EtlPipeline()
    return _instance


class EtlPipeline:
    """PDF 离线入库编排器

    后台任务引用由 _tasks 持有（防 GC）；状态与错误信息全部落注册表，
    前端轮询状态端点即可获知进度。
    """

    def __init__(self, db_path: str = None):
        self.db_router = DatabaseRouter(db_path)
        self.repo = self.db_router.pdf_documents
        self.mineru = MinerUService()
        self.vector_store = get_vector_store()
        self._tasks: Dict[str, asyncio.Task] = {}  # doc_id -> 后台任务

    @staticmethod
    def make_doc_id(filename: str) -> str:
        """文档级唯一标识：SHA-256(原始文件名) 完整 hex（两级哈希 ID 策略）"""
        return hashlib.sha256(filename.encode("utf-8")).hexdigest()

    def submit_pdf(self, filename: str, file_bytes: bytes) -> str:
        """提交 PDF 入库任务（立即返回 doc_id，后台执行解析/切分/入库）"""
        doc_id = self.make_doc_id(filename)
        self.repo.upsert(doc_id, filename, status="pending")

        task = asyncio.create_task(self._run_ingest(doc_id, filename, file_bytes))
        self._tasks[doc_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(doc_id, None))
        logger.info("已提交 PDF 入库任务 doc_id=%s filename=%s", doc_id, filename)
        return doc_id

    async def _run_ingest(self, doc_id: str, filename: str, file_bytes: bytes):
        """后台入库任务：状态机流转，任何环节失败落 failed + 错误信息"""
        try:
            self.repo.update_status(doc_id, "parsing")
            result = await self.mineru.parse_pdf(filename, file_bytes)

            self.repo.update_status(doc_id, "chunking")
            chunks = await asyncio.to_thread(
                chunk_document, result.markdown, doc_id, result.pages
            )
            if not chunks:
                raise RuntimeError("切分结果为空：文档无可入库的文本内容")

            self.repo.update_status(doc_id, "indexing")
            # 文档更新语义：先清空该 doc_id 全部旧 chunk，再写入新 chunk
            await self.vector_store.delete_by_doc_id(doc_id)
            await self.vector_store.upsert_chunks(chunks)

            self.repo.update_status(doc_id, "done", chunk_count=len(chunks))
            logger.info("PDF 入库完成 doc_id=%s chunks=%d", doc_id, len(chunks))
        except Exception as e:
            logger.error("PDF 入库失败 doc_id=%s: %s", doc_id, e)
            self.repo.update_status(doc_id, "failed", error_message=str(e))

    def get_status(self, doc_id: str) -> Optional[Dict]:
        """获取文档入库状态（注册表记录；不存在返回 None）"""
        return self.repo.get(doc_id)

    def list_documents(self) -> List[Dict]:
        """获取全部 PDF 文档注册记录（含状态与 chunk 数）"""
        return self.repo.list_all()

    async def delete_document(self, doc_id: str) -> bool:
        """删除文档：清空 Milvus 内该 doc_id 全部 chunk + 移除注册表记录"""
        await self.vector_store.delete_by_doc_id(doc_id)
        return self.repo.remove(doc_id)
