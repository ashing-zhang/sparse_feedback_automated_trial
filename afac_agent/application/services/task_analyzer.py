"""
运行指南：
- 本模块负责对数据集做轻量画像（dataset_profile），不直接运行。
- ExperimentController 会在每个任务开始时调用该分析器，将结果写入 AgentState 供策略决策使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from afac_agent.domain.models.datasets import ClassificationDataset, RecommendationDataset


@dataclass(frozen=True, slots=True)
class TaskAnalyzer:
    """数据集画像分析器。"""

    def analyze_classification(self, dataset: ClassificationDataset) -> dict[str, Any]:
        """生成分类任务的 dataset_profile。"""
        y = np.asarray(dataset.labels)
        train_idx = np.asarray(dataset.train_idx)
        test_idx = np.asarray(dataset.test_idx)

        y_train = y[train_idx] if train_idx.size else np.asarray([])
        unique, counts = np.unique(y_train, return_counts=True) if y_train.size else (np.asarray([]), np.asarray([]))
        dist = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist(), strict=False)}

        adjacency = dataset.adjacency
        attributes = dataset.attributes

        return {
            "task": "classification",
            "nodes": int(adjacency.shape[0]),
            "edges_nnz": int(adjacency.nnz),
            "attributes_shape": [int(x) for x in attributes.shape],
            "attributes_nnz": int(attributes.nnz),
            "train_size": int(train_idx.size),
            "test_size": int(test_idx.size),
            "train_label_distribution": dist,
        }

    def analyze_recommendation(self, dataset: RecommendationDataset) -> dict[str, Any]:
        """生成推荐任务的 dataset_profile。"""
        train = dataset.train
        test = dataset.test

        train_users = train["uid"].nunique() if "uid" in train.columns else None
        train_items = train["target_iid"].nunique() if "target_iid" in train.columns else None

        return {
            "task": "recommendation",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_unique_users": int(train_users) if train_users is not None else None,
            "train_unique_items": int(train_items) if train_items is not None else None,
            "users_rows": int(len(dataset.users)),
            "items_rows": int(len(dataset.items)),
        }
