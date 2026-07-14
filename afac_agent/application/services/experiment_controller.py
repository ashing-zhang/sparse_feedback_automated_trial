"""
运行指南：
- 本模块实现串行实验控制器（Observe→Diagnosis→Design→Execute→Evaluate→Memory Update→Next Decision），不直接运行。
- 由 ExperimentAgent 在 A1/A2 任务中分别调用本控制器，选择最佳 candidate 进入最终全量训练与预测。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any
import uuid

import numpy as np

from afac_agent.application.services.memory_system import MemorySystem
from afac_agent.application.services.task_analyzer import TaskAnalyzer
from afac_agent.application.services.splitters import random_split_row_indices, stratified_split_indices
from afac_agent.domain.models.agent_state import AgentState, TaskType
from afac_agent.domain.models.datasets import ClassificationDataset, RecommendationDataset
from afac_agent.domain.ports.experiment_policy import ExperimentPolicy
from afac_agent.domain.ports.experiment_store import ExperimentStore
from afac_agent.infrastructure.algorithms.factory import (
    build_classification_algorithm,
    build_recommendation_algorithm,
)
from afac_agent.infrastructure.config.schema import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExperimentController:
    """串行实验控制器。"""

    config: AppConfig
    policy: ExperimentPolicy
    store: ExperimentStore
    analyzer: TaskAnalyzer
    memory: MemorySystem

    def run_classification(self, dataset: ClassificationDataset) -> dict[str, Any]:
        """对分类任务执行闭环实验并返回最佳 candidate。"""
        profile = self.analyzer.analyze_classification(dataset)
        split = stratified_split_indices(
            indices=dataset.train_idx.tolist(),
            labels=np.asarray(dataset.labels),
            val_ratio=self.config.data.classification.val_ratio,
            seed=self.config.run.seed,
        )
        allowed = list(self.config.search_space.classification_candidates)
        mutation_rules = dict(self.config.search_space.classification_mutation_rules)

        def _run(candidate: dict[str, Any]) -> tuple[float, dict[str, Any]]:
            algo = build_classification_algorithm(candidate)
            result = algo.run(dataset, train_idx=split.train, val_idx=split.val)
            score = float(result.val_accuracy or 0.0)
            return score, {"val_accuracy": score}

        return self._run_task(
            task_type="classification",
            dataset_profile=profile,
            allowed_candidates=allowed,
            mutation_rules=mutation_rules,
            run_once=_run,
        )

    def run_recommendation(self, dataset: RecommendationDataset) -> dict[str, Any]:
        """对推荐任务执行闭环实验并返回最佳 candidate。"""
        profile = self.analyzer.analyze_recommendation(dataset)
        split = random_split_row_indices(
            row_count=len(dataset.train),
            val_ratio=self.config.data.recommendation.val_ratio,
            seed=self.config.run.seed,
        )
        allowed = list(self.config.search_space.recommendation_candidates)
        mutation_rules = dict(self.config.search_space.recommendation_mutation_rules)

        def _run(candidate: dict[str, Any]) -> tuple[float, dict[str, Any]]:
            algo = build_recommendation_algorithm(candidate)
            result = algo.run(dataset, train_row_idx=split.train, val_row_idx=split.val)
            score = float(result.val_ndcg_at_10 or 0.0)
            return score, {"val_ndcg_at_10": score}

        return self._run_task(
            task_type="recommendation",
            dataset_profile=profile,
            allowed_candidates=allowed,
            mutation_rules=mutation_rules,
            run_once=_run,
        )

    def _run_task(
        self,
        *,
        task_type: TaskType,
        dataset_profile: dict[str, Any],
        allowed_candidates: list[dict[str, Any]],
        mutation_rules: dict[str, Any],
        run_once: Any,
    ) -> dict[str, Any]:
        """通用串行实验循环。"""
        if not allowed_candidates:
            raise ValueError("allowed_candidates is empty")

        run_id = uuid.uuid4().hex[:8]
        total_rounds = int(self.config.controller.max_rounds or self.config.run.experiment.max_trials)
        failure_cases: list[str] = []
        local_history: list[dict[str, Any]] = []

        best_candidate: dict[str, Any] | None = None
        best_score = float("-inf")

        long_term = self.memory.get_long_term_task_records(task_type)
        for round_id in range(1, total_rounds + 1):
            rounds_left = total_rounds - round_id + 1
            history = _merge_history(long_term, local_history, limit=self.config.memory.recent_k)
            state = AgentState.from_context(
                task_type=task_type,
                dataset_profile=dataset_profile,
                experiment_history=history,
                best_solution=best_candidate,
                remaining_budget={
                    "rounds_left": rounds_left,
                    "total_rounds": total_rounds,
                    "consumed_rounds": round_id - 1,
                },
                failure_cases=failure_cases,
            )

            should_stop, stop_reason = self.policy.should_stop(state)
            if should_stop:
                logger.info("Early stop: task=%s reason=%s", task_type, stop_reason)
                break

            trial = self.policy.propose_next_trial(
                state,
                allowed_candidates=allowed_candidates,
                mutation_rules=mutation_rules,
            )
            candidate = dict(trial.candidate)
            logger.info(
                "Trial start: task=%s round=%d stage=%s candidate=%s",
                task_type,
                round_id,
                trial.stage,
                candidate,
            )

            try:
                score, metrics = run_once(candidate)
                improved = score > best_score
                if improved:
                    best_score = score
                    best_candidate = candidate
            except Exception as e:
                msg = f"trial_failed: {e}"
                failure_cases.append(msg)
                metrics = {"error": msg}
                score = float("-inf")
                improved = False

            record = {
                "time": _utc_now(),
                "event": "trial_completed",
                "run_id": run_id,
                "task": task_type,
                "trial_id": round_id,
                "stage": trial.stage,
                "reason": trial.reason,
                "source": trial.source,
                "candidate": candidate,
                **metrics,
                "improved": improved,
                "best_score": best_score,
            }
            local_history.append(record)
            self.store.append(record)

        if best_candidate is None:
            best_candidate = allowed_candidates[0]
        return best_candidate


def _utc_now() -> str:
    """返回 UTC 时间戳字符串。"""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _merge_history(
    long_term: list[dict[str, Any]],
    local: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """合并长期与本次历史，并限制长度。"""
    if limit <= 0:
        return list(local)
    merged = list(long_term[-limit:]) + list(local[-limit:])
    return merged[-limit:]

