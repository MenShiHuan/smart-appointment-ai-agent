"""
PDF 文档注册表数据访问对象

离线入库链路的状态机持久层：doc_id 为主键（SHA-256 文件名），
同名重传走 upsert 重置为 pending（文档更新语义）。
"""
from typing import List, Dict, Any, Optional
from ..base.session_manager import SessionManager
from ..models import utcnow, PdfDocument


class PdfDocumentRepository:
    """
    PDF 文档注册表数据访问对象

    职责：
    1. 文档入库状态机的持久化（pending/parsing/chunking/indexing/done/failed）
    2. 文档列表与状态查询
    3. 文档删除
    """

    def __init__(self, session_manager: SessionManager):
        """
        初始化 PDF 文档注册表数据仓库

        Args:
            session_manager: 会话管理器
        """
        self.session_manager = session_manager

    def upsert(self, doc_id: str, filename: str, status: str = "pending") -> bool:
        """
        插入或重置文档记录

        同名重传（doc_id 相同）→ 状态重置为 pending、清空计数与错误信息。

        Args:
            doc_id: 文档唯一标识（SHA-256 文件名）
            filename: 原始文件名
            status: 初始状态

        Returns:
            是否成功
        """
        with self.session_manager.session_scope() as session:
            document = session.query(PdfDocument).filter(
                PdfDocument.doc_id == doc_id
            ).first()
            if not document:
                session.add(PdfDocument(
                    doc_id=doc_id, original_filename=filename, status=status
                ))
            else:
                document.original_filename = filename
                document.status = status
                document.chunk_count = 0
                document.error_message = None
                document.updated_at = utcnow()
            return True

    def update_status(self, doc_id: str, status: str, chunk_count: Optional[int] = None,
                      error_message: Optional[str] = None) -> bool:
        """
        更新文档状态（进入非 failed 状态时自动清空错误信息）

        Args:
            doc_id: 文档唯一标识
            status: 新状态
            chunk_count: 入库 chunk 数（可选）
            error_message: 错误信息（可选，仅 failed 状态保留）

        Returns:
            是否成功
        """
        with self.session_manager.session_scope() as session:
            document = session.query(PdfDocument).filter(
                PdfDocument.doc_id == doc_id
            ).first()
            if not document:
                return False
            document.status = status
            if chunk_count is not None:
                document.chunk_count = chunk_count
            document.error_message = error_message if status == "failed" else None
            document.updated_at = utcnow()
            return True

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定文档注册记录

        Args:
            doc_id: 文档唯一标识

        Returns:
            记录字典，不存在返回 None
        """
        with self.session_manager.session_scope() as session:
            document = session.query(PdfDocument).filter(
                PdfDocument.doc_id == doc_id
            ).first()
            return self._to_dict(document) if document else None

    def list_all(self) -> List[Dict[str, Any]]:
        """
        获取全部文档注册记录（按更新时间倒序）

        Returns:
            记录字典列表
        """
        with self.session_manager.session_scope() as session:
            documents = session.query(PdfDocument).order_by(
                PdfDocument.updated_at.desc()
            ).all()
            return [self._to_dict(d) for d in documents]

    def remove(self, doc_id: str) -> bool:
        """
        删除文档注册记录

        Args:
            doc_id: 文档唯一标识

        Returns:
            是否成功
        """
        with self.session_manager.session_scope() as session:
            document = session.query(PdfDocument).filter(
                PdfDocument.doc_id == doc_id
            ).first()
            if not document:
                return False
            session.delete(document)
            return True

    @staticmethod
    def _to_dict(document: PdfDocument) -> Dict[str, Any]:
        """将文档注册对象转换为字典"""
        return {
            'doc_id': document.doc_id,
            'original_filename': document.original_filename,
            'status': document.status,
            'chunk_count': document.chunk_count,
            'error_message': document.error_message,
            'created_at': document.created_at,
            'updated_at': document.updated_at,
        }
