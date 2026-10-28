"""
LLM 输出 JSON 提取统一工具

收敛四处历史实现（input_parser 直接 loads / replan 手工剥围栏 /
memory_extractor 重试 loads / evals 正则取首段）为单一超集行为：
1. strip 后剥首尾 ``` 围栏（仅剥首尾，不做全局替换，
   避免破坏 JSON 字符串值内含 ``` 的病态情形）
2. 直接 json.loads（原 input_parser/memory_extractor 主路径）
3. 失败则贪婪正则取首个 {…}/[…] 片段再 loads（原 replan/evals 兜底路径）
4. 仍失败抛出 JSONDecodeError，由调用方按原样走修复重试/降级
"""
import json
import re

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


def extract_json(text: str):
    """从 LLM 输出中提取并解析 JSON（返回解析后的对象）"""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        # 剥首部围栏行（``` 或 ```json 等）与尾部 ```，仅首尾各一次
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        match = _JSON_BLOCK.search(cleaned)
        if not match:
            raise
        return json.loads(match.group(0))
