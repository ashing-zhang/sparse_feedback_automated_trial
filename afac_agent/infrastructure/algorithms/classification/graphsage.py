"""
运行指南：
- 本模块实现分类任务的 GraphSAGE 算法（Mean Aggregator），不直接运行。
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
    prepare_mean_adjacency,
    to_torch_sparse,
    train_node_classifier,
)

logger = logging.getLogger(__name__)


class _GraphSAGELayer(nn.Module):
    """GraphSAGE Mean 聚合层。"""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """初始化层参数。"""
        super().__init__()
        self.weight_self = nn.Parameter(torch.empty(in_dim, out_dim))
        self.weight_neigh = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight_self)
        nn.init.xavier_uniform_(self.weight_neigh)

    def forward(self, inputs: torch.Tensor, mean_adj: torch.Tensor) -> torch.Tensor:
        """执行一次 GraphSAGE Mean 聚合。"""
        self_part = linear_sparse_or_dense(inputs, self.weight_self)
        neigh_transformed = linear_sparse_or_dense(inputs, self.weight_neigh)
        neigh_part = torch.sparse.mm(mean_adj, neigh_transformed)
        return self_part + neigh_part + self.bias


class _GraphSAGEModel(nn.Module):
    """两层 GraphSAGE 节点分类模型。"""

    def __init__(
        self,
        *,
        mean_adjacency: torch.Tensor,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        """初始化 GraphSAGE 模型。"""
        super().__init__()
        self.register_buffer("mean_adjacency", mean_adjacency)
        self.layer1 = _GraphSAGELayer(input_dim, hidden_dim)
        self.layer2 = _GraphSAGELayer(hidden_dim, num_classes)
        self.dropout = float(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向计算全图节点 logits。"""
        hidden = self.layer1(inputs, self.mean_adjacency)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.layer2(hidden, self.mean_adjacency)


@dataclass(frozen=True, slots=True)
class GraphSAGEAlgorithm:
    """GraphSAGE Mean Aggregator（两层）。"""

    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    seed: int = 42
    patience: int = 5

    def run(
        self,
        dataset: ClassificationDataset,
        train_idx: list[int],
        val_idx: list[int],
    ):
        """训练并评估，同时对测试集生成预测。"""
        num_classes = infer_num_classes(dataset.labels, dataset.train_idx)
        mean_adjacency = prepare_mean_adjacency(dataset.adjacency)
        mean_adj_tensor = to_torch_sparse(mean_adjacency, torch.device("cpu"))

        model = _GraphSAGEModel(
            mean_adjacency=mean_adj_tensor,
            input_dim=int(dataset.attributes.shape[1]),
            hidden_dim=int(self.hidden_dim),
            num_classes=num_classes,
            dropout=float(self.dropout),
        )
        logger.info(
            "GraphSAGE initialized hidden_dim=%d dropout=%.3f lr=%.6f weight_decay=%.6f epochs=%d",
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
                patience=int(self.patience),
            ),
            algorithm_name="GraphSAGE",
        )
