"""
运行指南：
- 本文件为领域层端口（Protocol）定义，不直接运行。
- 基础设施层实现这些端口（从文件读取数据集）。
"""

from __future__ import annotations

from typing import Protocol

from afac_agent.domain.models.datasets import ClassificationDataset, RecommendationDataset


class DatasetLoader(Protocol):
    """数据加载端口。"""

    def load_classification(self) -> ClassificationDataset:
        """加载产品分类任务数据。"""

    def load_recommendation(self) -> RecommendationDataset:
        """加载产品推荐任务数据。"""

