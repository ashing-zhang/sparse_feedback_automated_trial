"""
运行指南：
- 本模块实现基于 LLM 的实验决策策略（Planner/Scientist/Engineer/Reviewer 委托），不直接运行。
- 当 LLM 不可用（禁用/缺钥/请求失败/输出不可解析）时，会自动回退到启发式策略以保持系统可运行。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

from afac_agent.application.services.llm_roles.role_agent import RoleAgent
from afac_agent.domain.models.agent_state import AgentState
from afac_agent.domain.models.trial_spec import TrialSpec
from afac_agent.domain.ports.experiment_policy import ExperimentPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMExperimentPolicy(ExperimentPolicy):
    """LLM 驱动的实验决策策略。"""

    planner: RoleAgent
    scientist: RoleAgent
    engineer: RoleAgent
    reviewer: RoleAgent
    improvement_threshold: float
    patience: int

    def propose_next_trial(
        self,
        state: AgentState,
        *,
        allowed_candidates: list[dict[str, Any]],
        mutation_rules: dict[str, Any],
    ) -> TrialSpec:
        """基于上下文提出下一轮要执行的 trial 配置。"""
        tried = {_canon(x.get("candidate")) for x in state.experiment_history if isinstance(x, dict)}

        planner_out = _safe_run(
            self.planner,
            {
                "state": _state_payload(state),
                "allowed_candidates": allowed_candidates,
                "mutation_rules": mutation_rules,
            },
        )
        if planner_out is None:
            stage = _heuristic_stage(state)
            candidate = _pick_next_untried(allowed_candidates, tried) or (state.best_solution or allowed_candidates[0])
            return TrialSpec(candidate=dict(candidate), stage=stage, reason="heuristic_fallback", source="heuristic")

        decision = str(planner_out.get("decision", "")).strip().lower()
        stop = bool(planner_out.get("stop", False))
        reason = str(planner_out.get("reason", "")).strip()
        if stop:
            candidate = state.best_solution or (allowed_candidates[0] if allowed_candidates else {})
            return TrialSpec(candidate=dict(candidate), stage="stabilize", reason=reason or "planner_stop", source="llm")

        stage = decision if decision in {"explore", "exploit", "stabilize"} else _heuristic_stage(state)

        scientist_out = _safe_run(
            self.scientist,
            {
                "state": _state_payload(state),
                "allowed_candidates": allowed_candidates,
                "mutation_rules": mutation_rules,
            },
        )

        engineer_out = _safe_run(
            self.engineer,
            {
                "state": _state_payload(state),
                "planner_decision": planner_out,
                "scientist_analysis": scientist_out or {},
                "allowed_candidates": allowed_candidates,
                "mutation_rules": mutation_rules,
            },
        )

        candidate = None
        fallback_used = True
        engineer_reason = "engineer_fallback"
        if engineer_out is not None:
            cand = engineer_out.get("candidate")
            if isinstance(cand, dict) and _is_candidate_allowed(cand, allowed_candidates):
                candidate = cand
                fallback_used = bool(engineer_out.get("fallback_used", False))
                engineer_reason = str(engineer_out.get("reason", "")).strip() or engineer_reason

        if candidate is None:
            candidate = _pick_next_untried(allowed_candidates, tried) or (state.best_solution or allowed_candidates[0])

        spec_reason = "; ".join([x for x in [reason, engineer_reason] if x])
        return TrialSpec(
            candidate=dict(candidate),
            stage="fallback" if fallback_used else stage,  # type: ignore[arg-type]
            reason=spec_reason or "ok",
            source="llm" if engineer_out is not None else "heuristic",
        )

    def should_stop(self, state: AgentState) -> tuple[bool, str]:
        """判断是否应提前停止并返回原因。"""
        heuristic = _heuristic_should_stop(state, improvement_threshold=self.improvement_threshold, patience=self.patience)
        if heuristic[0]:
            return heuristic

        reviewer_out = _safe_run(
            self.reviewer,
            {
                "state": _state_payload(state),
                "controller_config": {
                    "improvement_threshold": self.improvement_threshold,
                    "patience": self.patience,
                },
            },
        )
        if reviewer_out is None:
            return False, "reviewer_unavailable"
        return bool(reviewer_out.get("should_stop", False)), str(reviewer_out.get("reason", "")).strip()


def _state_payload(state: AgentState) -> dict[str, Any]:
    """将 AgentState 转为可序列化 payload。"""
    return {
        "task_type": state.task_type,
        "dataset_profile": state.dataset_profile,
        "experiment_history": state.experiment_history,
        "best_solution": state.best_solution,
        "remaining_budget": state.remaining_budget,
        "failure_cases": state.failure_cases,
    }


def _safe_run(agent: RoleAgent, payload: dict[str, Any]) -> dict[str, Any] | None:
    """安全执行 LLM 角色调用。"""
    try:
        return agent.run(payload)
    except Exception as e:
        logger.info("Role agent failed, fallback: %s", e)
        return None


def _canon(obj: Any) -> str:
    """将 candidate 规范化为稳定字符串用于去重。"""
    if not isinstance(obj, dict):
        return ""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _is_candidate_allowed(candidate: dict[str, Any], allowed: list[dict[str, Any]]) -> bool:
    """校验 candidate.kind 是否允许。"""
    kind = str(candidate.get("kind", "")).strip()
    if not kind:
        return False
    allowed_kinds = {str(x.get("kind", "")).strip() for x in allowed if isinstance(x, dict)}
    return kind in allowed_kinds


def _pick_next_untried(
    allowed: list[dict[str, Any]],
    tried: set[str],
) -> dict[str, Any] | None:
    """从 allowed_candidates 中选择一个未尝试的 candidate。"""
    for cand in allowed:
        if _canon(cand) not in tried:
            return cand
    return None


def _heuristic_stage(state: AgentState) -> str:
    """在无 LLM 时根据预算推断阶段。"""
    total = int(state.remaining_budget.get("total_rounds", 0) or 0)
    left = int(state.remaining_budget.get("rounds_left", 0) or 0)
    done = max(0, total - left)
    if total <= 0:
        return "explore"
    ratio = done / total
    if ratio < 0.3:
        return "explore"
    if ratio < 0.8:
        return "exploit"
    return "stabilize"


def _heuristic_should_stop(
    state: AgentState,
    *,
    improvement_threshold: float,
    patience: int,
) -> tuple[bool, str]:
    """基于近期提升与耐心参数的提前停止启发式。"""
    history = [x for x in state.experiment_history if isinstance(x, dict)]
    if not history:
        return False, "no_history"

    metric_key = "val_accuracy" if state.task_type == "classification" else "val_ndcg_at_10"
    scores = [float(x.get(metric_key) or 0.0) for x in history if metric_key in x]
    if len(scores) < 2:
        return False, "insufficient_scores"

    best = max(scores)
    last = scores[-1]
    improved = (last + improvement_threshold) >= best
    if improved:
        return False, "recent_improved"

    if len(scores) >= patience + 1:
        tail = scores[-(patience + 1) :]
        if max(tail[:-1]) - tail[-1] <= improvement_threshold:
            return True, "patience_exhausted"

    return False, "continue"

