"""
运行指南：
- 本文件为领域层数据结构定义，不直接运行。
- 由应用层在每轮实验中构造/更新 AgentState，并交由策略组件（如 LLM 驱动的决策器）使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TaskType = Literal["classification", "recommendation"]


@dataclass(frozen=True, slots=True)
class AgentState:
    """Agent 实验上下文（用于闭环优化）。"""

    task_type: TaskType
    dataset_profile: dict[str, Any]
    experiment_history: list[dict[str, Any]]
    best_solution: dict[str, Any] | None
    remaining_budget: dict[str, Any]
    failure_cases: list[str]

    @classmethod
    def from_context(
        cls,
        *,
        task_type: TaskType,
        dataset_profile: dict[str, Any],
        experiment_history: list[dict[str, Any]],
        best_solution: dict[str, Any] | None,
        remaining_budget: dict[str, Any],
        failure_cases: list[str],
    ) -> "AgentState":
        """从上下文构建 AgentState。"""
        return cls(
            task_type=task_type,
            dataset_profile=dict(dataset_profile),
            experiment_history=list(experiment_history),
            best_solution=dict(best_solution) if best_solution is not None else None,
            remaining_budget=dict(remaining_budget),
            failure_cases=list(failure_cases),
        )

