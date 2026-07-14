"""
运行指南：
- 本文件为领域层端口（Protocol）定义，不直接运行。
- 应用层会组合 Planner/Scientist/Engineer/Reviewer 等组件实现具体策略，并通过该端口暴露给编排器。
"""

from __future__ import annotations

from typing import Any, Protocol

from afac_agent.domain.models.agent_state import AgentState
from afac_agent.domain.models.trial_spec import TrialSpec


class ExperimentPolicy(Protocol):
    """实验决策策略端口。"""

    def propose_next_trial(
        self,
        state: AgentState,
        *,
        allowed_candidates: list[dict[str, Any]],
        mutation_rules: dict[str, Any],
    ) -> TrialSpec:
        """基于上下文提出下一轮要执行的 trial 配置。"""

    def should_stop(self, state: AgentState) -> tuple[bool, str]:
        """判断是否应提前停止并返回原因。"""

