"""
运行指南：
- 本模块实现轻量实验记忆系统（Short/Long Term），不直接运行。
- 默认基于 JsonlExperimentStore 的结构化记录进行检索；未来可替换为向量检索实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from afac_agent.domain.models.agent_state import TaskType
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

