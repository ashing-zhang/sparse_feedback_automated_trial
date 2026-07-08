"""
运行指南：
- 本文件为领域层端口（Protocol）定义，不直接运行。
- 基础设施层提供具体算法实现（如 Label Propagation、Logistic Regression、Popularity、Co-Occurrence 等）。
"""

from __future__ import annotations

from typing import Protocol

from afac_agent.domain.models.datasets import ClassificationDataset, RecommendationDataset
from afac_agent.domain.models.runs import ClassificationRunResult, RecommendationRunResult


class ClassificationAlgorithm(Protocol):
    """分类任务算法端口。"""

    def run(
        self,
        dataset: ClassificationDataset,
        train_idx: list[int],
        val_idx: list[int],
    ) -> ClassificationRunResult:
        """训练并评估（如提供 val_idx），同时对测试集生成预测。"""


class RecommendationAlgorithm(Protocol):
    """推荐任务算法端口。"""

    def run(
        self,
        dataset: RecommendationDataset,
        train_row_idx: list[int],
        val_row_idx: list[int],
    ) -> RecommendationRunResult:
        """训练并评估（如提供 val_row_idx），同时对测试集生成预测。"""

