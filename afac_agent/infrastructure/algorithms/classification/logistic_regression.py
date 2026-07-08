"""
运行指南：
- 本模块实现分类任务的稀疏特征 Logistic Regression 基线算法，不直接运行。
- 由 Experiment Agent 按配置创建该算法并执行 run()。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from afac_agent.application.services.metrics import accuracy
from afac_agent.domain.models.datasets import ClassificationDataset
from afac_agent.domain.models.predictions import A1PredictionRow
from afac_agent.domain.models.runs import ClassificationRunResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LogisticRegressionAlgorithm:
    """Logistic Regression（仅用节点特征，不使用图结构）。"""

    c: float
    max_iter: int

    def run(
        self,
        dataset: ClassificationDataset,
        train_idx: list[int],
        val_idx: list[int],
    ) -> ClassificationRunResult:
        """训练并评估，同时对测试集生成预测。"""
        x = dataset.attributes
        y = np.asarray(dataset.labels, dtype=int)

        if not train_idx:
            raise ValueError("empty train_idx")

        model = LogisticRegression(
            C=float(self.c),
            max_iter=int(self.max_iter),
            n_jobs=1,
            solver="saga",
            multi_class="multinomial",
        )
        model.fit(x[train_idx], y[train_idx])

        val_acc: float | None = None
        if val_idx:
            pred_val = model.predict(x[val_idx]).astype(int).tolist()
            y_true = [int(y[i]) for i in val_idx]
            val_acc = accuracy(y_true, pred_val)
            logger.info("LogisticRegression val_accuracy=%.6f", val_acc)

        test_pred = model.predict(x[dataset.test_idx]).astype(int).tolist()
        test_predictions = [
            A1PredictionRow(test_idx=int(idx), label=int(label))
            for idx, label in zip(dataset.test_idx.tolist(), test_pred, strict=False)
        ]
        return ClassificationRunResult(val_accuracy=val_acc, test_predictions=test_predictions)

