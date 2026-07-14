"""
运行指南：
- 本文件为领域层端口（Protocol）定义，不直接运行。
- 基础设施层实现实验记录存储（默认 JSONL）。
"""

from __future__ import annotations

from typing import Any, Protocol


class ExperimentStore(Protocol):
    """实验记录存储端口。"""

    def append(self, record: dict[str, Any]) -> None:
        """追加一条实验记录。"""

    def read_recent(self, limit: int) -> list[dict[str, Any]]:
        """读取最近若干条实验记录（按写入顺序从新到旧）。"""
