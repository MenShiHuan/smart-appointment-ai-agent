"""
MinerU 云文档解析服务客户端（官方 OpenAPI v4：https://mineru.net/api/v4）

本地不部署 MinerU；凭证为 .env 的 MINERU_TOKEN（Authorization: Bearer）。
单文件上传解析流程（v4 三端点）：
1. POST /file-urls/batch            申请预签名上传地址（files: [{name, is_ocr}]）
2. PUT  <presigned_url>             直传文件字节（预签名 URL，无需鉴权头）
3. POST /extract/task/batch         触发解析（单文件 batch；model_version=vlm 为精准解析模式）
4. GET  /extract-results/batch/{id} 轮询至 done/failed，取 md_content 与 content_list

解析参数：is_ocr=True（扫描件/公式识别）、language=ch、enable_formula=True、
model_version 可配（vlm=精准 / pipeline=普通，见 config/vector_config.py）。
页眉/页脚/水印过滤由 MinerU 服务端完成，应用侧只消费其产物。
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import requests

from config import vector_config as vc

logger = logging.getLogger(__name__)

UPLOAD_PATH = vc.MINERU_UPLOAD_PATH
PARSE_PATH = vc.MINERU_PARSE_PATH
RESULT_PATH = vc.MINERU_RESULT_PATH

# 批次/单文件完成与失败状态（v4 状态机：pending → running → converting → done/failed）
DONE_STATES = {"done", "success", "finished", "completed", "succeeded"}
FAILED_STATES = {"failed", "error", "cancelled"}


@dataclass
class MinerUResult:
    """MinerU 解析产物（应用侧标准化结构）"""
    markdown: str
    pages: Optional[List[int]] = None  # 与 markdown 行对齐的页码（1 起）；缺省 None
    raw: Dict[str, Any] = field(default_factory=dict)


class MinerUService:
    """MinerU 云 API 客户端：单文件上传 → 触发解析 → 轮询结果 → 标准化产物"""

    def __init__(self, base_url: str = None, token: str = None):
        self.base_url = (base_url or vc.MINERU_BASE_URL).rstrip("/")
        self.token = token or vc.MINERU_TOKEN

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _check_code(payload: Dict[str, Any], action: str) -> None:
        """云 API 网关约定：code==0 表示成功，否则抛结构化错误"""
        if payload.get("code") not in (0, None):
            raise RuntimeError(
                f"MinerU {action} 失败: code={payload.get('code')} "
                f"{payload.get('msg') or payload.get('message') or str(payload)[:200]}"
            )

    async def parse_pdf(self, filename: str, file_bytes: bytes) -> MinerUResult:
        """解析 PDF（阻塞型 HTTP 调用全部放线程池，避免阻塞事件循环）"""
        batch_id = await asyncio.to_thread(self._submit_sync, filename, file_bytes)
        payload = await asyncio.to_thread(self._poll_sync, batch_id)
        return await asyncio.to_thread(self._extract_result, payload)

    # ============ 同步实现（经 to_thread 调用） ============

    def _submit_sync(self, filename: str, file_bytes: bytes) -> str:
        """单文件上传解析：申请预签名地址 → 直传 → 触发解析，返回 batch_id"""
        if not self.token:
            raise RuntimeError("MINERU_TOKEN 未配置：请在 .env 中填写 MinerU 云 API 凭证")

        # 1. 申请预签名上传地址
        resp = requests.post(
            f"{self.base_url}{UPLOAD_PATH}",
            json={"files": [{"name": filename, "is_ocr": True}]},
            headers=self._headers(),
            timeout=vc.MINERU_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._check_code(payload, "申请上传地址")
        file_urls = (payload.get("data") or {}).get("file_urls") or []
        if not file_urls or not file_urls[0].get("url"):
            raise RuntimeError(f"MinerU 上传地址响应缺少 url: {str(payload)[:200]}")
        presigned_url = file_urls[0]["url"]

        # 2. 直传文件字节（预签名 URL 自身含凭证，PUT 无需鉴权头）
        put_resp = requests.put(presigned_url, data=file_bytes, timeout=vc.MINERU_TIMEOUT)
        put_resp.raise_for_status()

        # 3. 触发解析（单文件 batch；vlm=精准解析模式）
        resp = requests.post(
            f"{self.base_url}{PARSE_PATH}",
            json={
                "files": [{"url": presigned_url, "is_ocr": True}],
                "enable_formula": True,
                "language": "ch",
                "layout_model": "doclayout_yolo",
                "model_version": vc.MINERU_MODEL_VERSION,
            },
            headers=self._headers(),
            timeout=vc.MINERU_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._check_code(payload, "触发解析")
        batch_id = (payload.get("data") or {}).get("batch_id")
        if not batch_id:
            raise RuntimeError(f"MinerU 解析响应缺少 batch_id: {str(payload)[:200]}")
        logger.info("MinerU 解析任务已提交 batch_id=%s（%s 模式）",
                    batch_id, vc.MINERU_MODEL_VERSION)
        return str(batch_id)

    def _poll_sync(self, batch_id: str) -> Dict[str, Any]:
        """轮询批次结果至完成/失败/超时，返回结果载荷"""
        deadline = time.monotonic() + vc.MINERU_MAX_WAIT
        while True:
            resp = requests.get(
                f"{self.base_url}{RESULT_PATH.format(batch_id=batch_id)}",
                headers=self._headers(),
                timeout=vc.MINERU_TIMEOUT,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._check_code(payload, "查询结果")

            state = self._batch_state(payload)
            if state in DONE_STATES:
                return payload
            if state in FAILED_STATES:
                raise RuntimeError(f"MinerU 解析失败: {state} {str(payload)[:200]}")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"MinerU 解析超时（>{vc.MINERU_MAX_WAIT}s）batch_id={batch_id}"
                )
            time.sleep(vc.MINERU_POLL_INTERVAL)

    @staticmethod
    def _batch_state(payload: Dict[str, Any]) -> str:
        """批次整体状态：v4 结果载荷 data.extract_result 为单文件列表，逐项含 state"""
        data = payload.get("data") or {}
        results = data.get("extract_result")
        if isinstance(results, list) and results:
            states = {str(item.get("state", "")).lower() for item in results}
            if states & FAILED_STATES:
                return "failed"
            if states and states <= DONE_STATES:
                return "done"
            return "running"  # 有结果列表但未完成 → 继续等待
        return str(data.get("state") or payload.get("state") or "").lower()

    def _extract_result(self, payload: Dict[str, Any]) -> MinerUResult:
        """从批次结果提取 Markdown 与 layout 元数据（多字段名容忍）"""
        results = (payload.get("data") or {}).get("extract_result") or []
        if isinstance(results, dict):
            results = [results]
        if not results:
            raise RuntimeError(f"MinerU 返回中未找到解析结果: {str(payload)[:300]}")

        item = results[0]
        if str(item.get("state", "")).lower() in FAILED_STATES:
            raise RuntimeError(f"MinerU 单文件解析失败: {str(item)[:300]}")

        markdown = item.get("md_content") or item.get("markdown") or ""
        if not markdown:
            raise RuntimeError(f"MinerU 结果缺少 md_content: {str(item)[:300]}")

        content_list = item.get("content_list")
        pages = self._pages_from_content_list(content_list, markdown)
        return MinerUResult(markdown=markdown, pages=pages, raw=payload)

    @staticmethod
    def _pages_from_content_list(content_list, markdown: str) -> Optional[List[int]]:
        """content_list 逐块 page_idx → 与 markdown 行对齐的页码列表（1 起）"""
        if not isinstance(content_list, list) or not content_list:
            return None
        pages: List[int] = []
        for block in content_list:
            if not isinstance(block, dict):
                continue
            text = block.get("text") or ""
            page_idx = block.get("page_idx")
            if page_idx is None:
                page_idx = block.get("page")
            if page_idx is None:
                continue
            line_count = text.count("\n") + 1
            pages.extend([int(page_idx) + 1] * line_count)
        # 行数不足时返回 None，由 chunker 兜底为全 0
        return pages if len(pages) >= markdown.count("\n") + 1 else None
