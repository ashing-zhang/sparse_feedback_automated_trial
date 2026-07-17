"""
运行指南：
- 本模块实现分类任务的 Graph Transformer 算法（基于 Transformer 的图注意力），不直接运行。
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
    prepare_gcn_adjacency,
    to_torch_sparse,
    train_node_classifier,
)

logger = logging.getLogger(__name__)


class _GraphTransformerLayer(nn.Module):
    """Graph Transformer 层：使用自注意力机制聚合邻居信息。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = dropout

    def forward(self, inputs: torch.Tensor, adj_mask: torch.Tensor | None = None) -> torch.Tensor:
        """执行一次 Graph Transformer 层计算。"""
        residual = inputs
        attn_output, _ = self.self_attn(inputs, inputs, inputs, attn_mask=adj_mask)
        hidden = self.norm1(residual + F.dropout(attn_output, p=self.dropout, training=self.training))
        residual = hidden
        ffn_output = self.ffn(hidden)
        return self.norm2(residual + F.dropout(ffn_output, p=self.dropout, training=self.training))


class _GraphTransformerModel(nn.Module):
    """Graph Transformer 节点分类模型。"""

    def __init__(
        self,
        *,
        adjacency: torch.Tensor,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([_GraphTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = float(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向计算全图节点 logits。"""
        hidden = self.input_proj(inputs)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        adj_mask = self._build_adj_mask()
        for layer in self.layers:
            hidden = layer(hidden, adj_mask)

        return self.classifier(hidden)

    def _build_adj_mask(self) -> torch.Tensor | None:
        """构建注意力掩码：仅允许节点关注自身和邻居。"""
        adj = self.adjacency.coalesce()
        n_nodes = adj.shape[0]
        mask = torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=adj.device)
        mask[adj.indices()[0], adj.indices()[1]] = True
        mask[torch.arange(n_nodes), torch.arange(n_nodes)] = True
        return ~mask


@dataclass(frozen=True, slots=True)
class GraphTransformerAlgorithm:
    """Graph Transformer 节点分类算法。"""

    hidden_dim: int
    num_layers: int
    num_heads: int
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
        adjacency = prepare_gcn_adjacency(dataset.adjacency)
        adj_tensor = to_torch_sparse(adjacency, torch.device("cpu"))

        model = _GraphTransformerModel(
            adjacency=adj_tensor,
            input_dim=int(dataset.attributes.shape[1]),
            hidden_dim=int(self.hidden_dim),
            num_classes=num_classes,
            num_layers=int(self.num_layers),
            num_heads=int(self.num_heads),
            dropout=float(self.dropout),
        )
        logger.info(
            "GraphTransformer initialized hidden_dim=%d num_layers=%d num_heads=%d dropout=%.3f lr=%.6f weight_decay=%.6f epochs=%d",
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
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
            algorithm_name="GraphTransformer",
        )