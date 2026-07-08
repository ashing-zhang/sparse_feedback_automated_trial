"""
运行指南：
- 推荐任务算法的通用工具函数，不直接运行。
"""

from __future__ import annotations

from collections.abc import Iterable


def parse_item_seq(value: str) -> list[str]:
    """解析 item_seq_* 字段为 item 列表。"""
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        return []
    items = []
    for part in raw.split(","):
        item = part.strip().strip('"').strip("'")
        if item:
            items.append(item)
    return items


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    """对序列去重并保持顺序。"""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

