"""
运行指南：
- 本文件为领域层端口（Protocol）定义，不直接运行。
- 基础设施层实现官方提交文件写出、校验与打包。
"""

from __future__ import annotations

from typing import Protocol

from afac_agent.domain.models.predictions import A1PredictionRow, A2PredictionRow


class SubmissionWriter(Protocol):
    """提交文件写出端口。"""

    def write_a1(self, rows: list[A1PredictionRow]) -> str:
        """写出 A1.csv，返回文件路径。"""

    def write_a2(self, rows: list[A2PredictionRow]) -> str:
        """写出 A2.csv，返回文件路径。"""

    def write_zip(self, a1_path: str, a2_path: str) -> str:
        """将 A1/A2 打包为 prediction.zip，返回文件路径。"""

