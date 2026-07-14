"""
运行指南：
- 本文件为领域层数据结构定义，不直接运行。
- TrialSpec 描述“一轮实验要执行的配置”，用于记录与复现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

DecisionStage = Literal["explore", "exploit", "stabilize", "fallback"]


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """单轮实验配置描述。"""

    candidate: dict[str, Any]
    stage: DecisionStage
    reason: str
    source: str

