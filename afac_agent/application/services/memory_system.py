"""
运行指南：
- 本模块实现轻量实验记忆系统（Short/Long Term），不直接运行。
- 默认基于 JsonlExperimentStore 的结构化记录进行检索；未来可替换为向量检索实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from afac_agent.domain.models.agent_state import AgentState, TaskType
from afac_agent.domain.ports.experiment_store import ExperimentStore
from afac_agent.infrastructure.config.schema import MemoryConfig


@dataclass(frozen=True, slots=True)
class MemorySystem:
    """实验记忆系统。"""

    store: ExperimentStore
    config: MemoryConfig

    def get_recent_task_records(self, task_type: TaskType) -> list[dict[str, Any]]:
        """读取最近的任务实验记录（Short Term）。"""
        records = self.store.read_recent(self.config.recent_k)
        return [r for r in records if r.get("task") == task_type]

    def get_long_term_task_records(self, task_type: TaskType) -> list[dict[str, Any]]:
        """读取长期任务实验记录（Long Term）。"""
        records = self.store.read_recent(self.config.long_term_k)
        return [r for r in records if r.get("task") == task_type]

    def build_state(
        self,
        *,
        task_type: TaskType,
        dataset_profile: dict[str, Any],
        experiment_history: list[dict[str, Any]],
        best_solution: dict[str, Any] | None,
        remaining_budget: dict[str, Any],
        failure_cases: list[str],
    ) -> AgentState:
        """构建 AgentState。"""
        return AgentState(
            task_type=task_type,
            dataset_profile=dict(dataset_profile),
            experiment_history=list(experiment_history),
            best_solution=dict(best_solution) if best_solution is not None else None,
            remaining_budget=dict(remaining_budget),
            failure_cases=list(failure_cases),
        )

