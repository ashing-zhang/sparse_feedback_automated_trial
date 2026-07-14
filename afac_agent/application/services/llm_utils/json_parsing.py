"""
运行指南：
- 本模块提供 LLM 输出 JSON 的解析工具，不直接运行。
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    """从文本中解析 JSON object。"""
    raw = text.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return _loads_object(raw)

    fenced = _extract_code_fence(raw)
    if fenced is not None:
        return _loads_object(fenced)

    sliced = _slice_first_object(raw)
    if sliced is not None:
        return _loads_object(sliced)

    raise ValueError("cannot parse json object from llm output")


def _loads_object(raw: str) -> dict[str, Any]:
    """加载并校验为 dict。"""
    obj: Any = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("json must be an object")
    return obj


def _extract_code_fence(text: str) -> str | None:
    """提取 ```json ... ``` 或 ``` ... ``` 代码块内容。"""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def _slice_first_object(text: str) -> str | None:
    """尝试从文本中截取第一个花括号对象。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return None

