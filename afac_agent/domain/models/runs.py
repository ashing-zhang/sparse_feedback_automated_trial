"""
运行指南：
- 本文件为领域层结构定义，不直接运行。
- 用于承载“单轮实验”的输出（验证指标与对测试集的预测）。
"""

from __future__ import annotations

from dataclasses import dataclass

from afac_agent.domain.models.predictions import A1PredictionRow, A2PredictionRow


@dataclass(frozen=True, slots=True)
class ClassificationRunResult:
    """分类任务单轮实验结果。"""

    val_accuracy: float | None
    test_predictions: list[A1PredictionRow]


@dataclass(frozen=True, slots=True)
class RecommendationRunResult:
    """推荐任务单轮实验结果。"""

    val_ndcg_at_10: float | None
    test_predictions: list[A2PredictionRow]

