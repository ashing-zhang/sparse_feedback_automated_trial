"""
运行指南：
- 本文件为领域层预测输出结构定义，不直接运行。
- 基础设施层负责把预测写为官方要求的 A1.csv / A2.csv，并打包为 prediction.zip。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class A1PredictionRow:
    """产品分类任务的单行预测结果。"""

    test_idx: int
    label: int


@dataclass(frozen=True, slots=True)
class A2PredictionRow:
    """产品推荐任务的单行预测结果。"""

    uid: str
    prediction: list[str]

