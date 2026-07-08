"""
运行指南：
- 本模块提供评估指标计算函数，不直接运行。
- 由算法实现与实验控制器调用，用于得到 Accuracy 与 NDCG@10 等反馈信号。
"""

from __future__ import annotations

import math
from typing import Sequence


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """计算分类准确率。"""
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred, strict=False) if int(a) == int(b))
    return float(correct) / float(len(y_true))


def ndcg_at_k_single_target(
    targets: Sequence[str],
    predictions: Sequence[Sequence[str]],
    k: int,
) -> float:
    """计算 NDCG@K（每个样本仅 1 个目标 item）。"""
    if len(targets) == 0:
        return 0.0
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions length mismatch")

    total = 0.0
    for target, pred_list in zip(targets, predictions, strict=False):
        score = 0.0
        for rank, item in enumerate(list(pred_list)[:k], start=1):
            if item == target:
                score = 1.0 / math.log2(rank + 1.0)
                break
        total += score

    return total / float(len(targets))

