"""
文本切分器（RAG 重构）：结构感知 + 递归字符 + 滑动窗口

1. 结构感知切分：按 Markdown 标题层级（#~######）把文档拆成父块（章节），
   每块记录章节标题（层级路径，如「第一章 / 1.1 概述」），标题行计入父块首行；
2. 递归字符切分：父块文本按分隔符优先级（段落 → 换行 → 句末标点 → 逗号 → 空格）
   递归切分到目标长度（CHUNK_SIZE_TOKENS 经 CHARS_PER_TOKEN 换算为字符数）；
3. 滑动窗口：相邻窗口重叠 CHUNK_OVERLAP_TOKENS 换算的字符数（窗口尾部回卷）。

输出 chunk 列表：{doc_id, page_num, chapter_title, seq, content}
- seq 为文档内递增序号（从 1 起，与 chunk_id 编码 {doc_id}_{seq} 一致）
- page_num 为该 chunk 首字符所在页码（MinerU layout 逐行页码，缺省 0）
"""
import re
from typing import List, Dict, Optional, Tuple

from config import vector_config as vc

# 递归切分分隔符：从强到弱（段落 → 换行 → 句末标点 → 从句标点 → 空格）
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ", ", " "]

# Markdown 标题行（1~6 级）
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _chars_for_tokens(tokens: int) -> int:
    """token 数 → 字符数（中文 1 字 ≈ 1 token，CHARS_PER_TOKEN 可配）"""
    return int(tokens * vc.CHARS_PER_TOKEN)


def _split_by_max(text: str, max_chars: int) -> List[str]:
    """递归字符切分：在 max_chars 窗口内找最后一个分隔符切割；无分隔符则硬切"""
    if not text or max_chars <= 0:
        return [text] if text else []
    pieces = []
    while len(text) > max_chars:
        cut = -1
        for sep in SEPARATORS:
            pos = text.rfind(sep, 0, max_chars)
            if pos > cut:
                cut = pos + len(sep)
        if cut <= 0:
            cut = max_chars
        pieces.append(text[:cut])
        text = text[cut:]
    if text:
        pieces.append(text)
    return pieces


def _split_sections(lines: List[str], pages: List[int]) -> List[Dict]:
    """结构感知切分：按标题层级拆父块，章节标题为层级路径"""
    sections = []
    title_stack: List[Tuple[int, str]] = []  # [(level, title)]
    current = {"title": "", "lines": [], "pages": []}

    for line, page in zip(lines, pages):
        m = HEADING_RE.match(line.strip())
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            while title_stack and title_stack[-1][0] >= level:
                title_stack.pop()
            title_stack.append((level, title))
            # 标题行开启新父块（标题文本计入内容，利于 BM25 命中）
            sections.append(current)
            current = {
                "title": " / ".join(t for _, t in title_stack),
                "lines": [line],
                "pages": [page],
            }
        else:
            current["lines"].append(line)
            current["pages"].append(page)
    sections.append(current)
    # 过滤空父块（无标题且无正文内容）
    return [s for s in sections if s["title"] or any(l.strip() for l in s["lines"])]


def _split_section(section: Dict, target_chars: int, overlap_chars: int) -> List[Tuple[str, int]]:
    """父块内部：递归字符切分 + 滑动窗口合并，返回 [(chunk_text, 起始页码)]"""
    lines, pages = section["lines"], section["pages"]
    text = "\n".join(lines)

    # 字符偏移 → 页码映射（MinerU 逐行页码；+1 计入换行符）
    page_at: List[int] = []
    for line, page in zip(lines, pages):
        page_at.extend([page] * (len(line) + 1))
    if len(page_at) < len(text) + 1:
        page_at.extend([page_at[-1] if page_at else 0] * (len(text) + 1 - len(page_at)))
    page_at = page_at[: len(text) + 1]

    pieces: List[Tuple[str, int]] = []
    offset = 0
    for piece_text in _split_by_max(text, target_chars):
        pieces.append((piece_text, page_at[offset] if offset < len(page_at) else 0))
        offset += len(piece_text)

    return _merge_windows(pieces, target_chars, overlap_chars)


def _merge_windows(pieces: List[Tuple[str, int]], target_chars: int, overlap_chars: int) -> List[Tuple[str, int]]:
    """滑动窗口合并：逐片累积至 target_chars 上限；新窗口从上一窗口尾部回卷 overlap_chars"""
    windows = []
    buffer_text = ""
    buffer_page = 0
    for text, page in pieces:
        if not buffer_text:
            buffer_text, buffer_page = text, page
            continue
        if len(buffer_text) + len(text) <= target_chars:
            buffer_text += text
            continue
        windows.append((buffer_text, buffer_page))
        if overlap_chars > 0:
            tail = buffer_text[-overlap_chars:]
            # 尾巴 + 新片仍超上限时裁短尾巴（注意 carry<=0 时不能切片，否则 [-0:] 取整段）
            if len(tail) + len(text) > target_chars:
                carry = target_chars - len(text)
                tail = tail[-carry:] if carry > 0 else ""
            if tail:
                buffer_text, buffer_page = tail, buffer_page
            else:
                buffer_text, buffer_page = "", 0
        else:
            buffer_text, buffer_page = "", 0
        if buffer_text:
            buffer_text += text
        else:
            buffer_text, buffer_page = text, page
    if buffer_text.strip():
        windows.append((buffer_text, buffer_page))
    return windows


def chunk_document(markdown: str, doc_id: str, pages: Optional[List[int]] = None) -> List[Dict]:
    """文档切分入口：结构感知父块 → 递归字符 + 滑动窗口子 chunk

    Args:
        markdown: MinerU 输出的 Markdown 全文
        doc_id: 文档唯一标识（SHA-256 文件名）
        pages: 与 markdown 按行对齐的页码列表（MinerU layout 提供，1 起）；
               缺省或长度不足时全部按 0 处理（无页码元数据）
    """
    target_chars = _chars_for_tokens(vc.CHUNK_SIZE_TOKENS)
    overlap_chars = _chars_for_tokens(vc.CHUNK_OVERLAP_TOKENS)

    lines = markdown.split("\n")
    if not pages or len(pages) < len(lines):
        pages = [0] * len(lines)

    chunks = []
    seq = 0
    for section in _split_sections(lines, pages):
        for text, page in _split_section(section, target_chars, overlap_chars):
            content = text.strip()
            if not content:
                continue
            seq += 1
            chunks.append({
                "doc_id": doc_id,
                "page_num": int(page or 0),
                "chapter_title": section["title"],
                "seq": seq,
                "content": content[:vc.CONTENT_MAX_CHARS],
            })
    return chunks
