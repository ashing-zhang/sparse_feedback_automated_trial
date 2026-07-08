"""
运行指南：
- 本模块实现分类任务的 Label Propagation 基线算法，不直接运行。
- 由 Experiment Agent 按配置创建该算法并执行 run()。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from scipy.sparse import csr_matrix

from afac_agent.application.services.metrics import accuracy
from afac_agent.domain.models.datasets import ClassificationDataset
from afac_agent.domain.models.predictions import A1PredictionRow
from afac_agent.domain.models.runs import ClassificationRunResult

logger = logging.getLogger(__name__)


def _row_normalize(mat: csr_matrix) -> csr_matrix:
    """对 CSR 矩阵做行归一化。"""
    row_sum = np.asarray(mat.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    inv[nonzero] = 1.0 / row_sum[nonzero]
    return mat.multiply(inv[:, None]).tocsr()


@dataclass(frozen=True, slots=True)
class LabelPropagationAlgorithm:
    """Label Propagation（基于图传播的半监督基线）。"""

    alpha: float
    max_iter: int

    def run(
        self,
        dataset: ClassificationDataset,
        train_idx: list[int],
        val_idx: list[int],
    ) -> ClassificationRunResult:
        """训练并评估，同时对测试集生成预测。"""
        labels = np.asarray(dataset.labels, dtype=int)
        num_nodes = labels.shape[0]

        known_idx = np.asarray(train_idx, dtype=int)
        y_known = labels[known_idx]
        if y_known.size == 0:
            raise ValueError("empty train_idx")
        num_classes = int(np.max(y_known)) + 1

        adj = dataset.adjacency
        adj = (adj + adj.transpose()).tocsr()
        p = _row_normalize(adj)

        y0 = np.zeros((num_nodes, num_classes), dtype=np.float32)
        y0[known_idx, y_known] = 1.0
        f = y0.copy()

        alpha = float(self.alpha)
        for _ in range(int(self.max_iter)):
            f = alpha * (p @ f) + (1.0 - alpha) * y0
            f[known_idx] = y0[known_idx]

        pred_all = np.argmax(f, axis=1).astype(int)

        val_acc: float | None = None
        if val_idx:
            y_true = [int(labels[i]) for i in val_idx]
            y_pred = [int(pred_all[i]) for i in val_idx]
            val_acc = accuracy(y_true, y_pred)
            logger.info("LabelPropagation val_accuracy=%.6f", val_acc)

        test_predictions = [A1PredictionRow(test_idx=int(i), label=int(pred_all[int(i)])) for i in dataset.test_idx]
        return ClassificationRunResult(val_accuracy=val_acc, test_predictions=test_predictions)

