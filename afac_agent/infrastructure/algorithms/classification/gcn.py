"""
运行指南：
- 本模块实现分类任务的 GCN 算法，不直接运行。
- 由 Experiment Agent 按配置创建该算法并执行 run()。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from afac_agent.domain.models.datasets import ClassificationDataset
from afac_agent.infrastructure.algorithms.classification.gnn_common import (
    GNNTrainingConfig,
    infer_num_classes,
    linear_sparse_or_dense,
    prepare_gcn_adjacency,
    to_torch_sparse,
    train_node_classifier,
)

logger = logging.getLogger(__name__)


class _GCNLayer(nn.Module):
    """单层 GCN 图卷积。"""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """初始化图卷积参数。"""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """执行一次图卷积传播。"""
        support = linear_sparse_or_dense(inputs, self.weight)
        return torch.sparse.mm(adjacency, support) + self.bias


class _GCNModel(nn.Module):
    """两层 GCN 节点分类模型。"""

    def __init__(
        self,
        *,
        adjacency: torch.Tensor,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        """初始化 GCN 模型。"""
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.layer1 = _GCNLayer(input_dim, hidden_dim)
        self.layer2 = _GCNLayer(hidden_dim, num_classes)
        self.dropout = float(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向计算全图节点 logits。"""
        hidden = self.layer1(inputs, self.adjacency)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.layer2(hidden, self.adjacency)


@dataclass(frozen=True, slots=True)
class GCNAlgorithm:
    """GCN（基于归一化邻接矩阵的两层图卷积）。"""

    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    seed: int = 42

    def run(
        self,
        dataset: ClassificationDataset,
        train_idx: list[int],
        val_idx: list[int],
    ):
        """训练并评估，同时对测试集生成预测。"""
        num_classes = infer_num_classes(dataset.labels, dataset.train_idx)
        normalized_adjacency = prepare_gcn_adjacency(dataset.adjacency)
        adjacency_tensor = to_torch_sparse(normalized_adjacency, torch.device("cpu"))

        model = _GCNModel(
            adjacency=adjacency_tensor,
            input_dim=int(dataset.attributes.shape[1]),
            hidden_dim=int(self.hidden_dim),
            num_classes=num_classes,
            dropout=float(self.dropout),
        )
        logger.info(
            "GCN initialized hidden_dim=%d dropout=%.3f lr=%.6f weight_decay=%.6f epochs=%d",
            self.hidden_dim,
            self.dropout,
            self.learning_rate,
            self.weight_decay,
            self.epochs,
        )
        return train_node_classifier(
            model,
            features=dataset.attributes,
            labels=np.asarray(dataset.labels, dtype=int),
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=np.asarray(dataset.test_idx, dtype=int),
            config=GNNTrainingConfig(
                epochs=int(self.epochs),
                learning_rate=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
                seed=int(self.seed),
            ),
            algorithm_name="GCN",
        )
