"""
运行指南：
- 本模块实现“串行实验 Agent”的核心编排逻辑，不直接运行。
- 入口模块会组装依赖并调用 ExperimentAgent.run_all() 来生成最终 prediction.zip。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import random

import numpy as np

from afac_agent.application.services.splitters import random_split_row_indices, stratified_split_indices
from afac_agent.domain.ports.dataset_loader import DatasetLoader
from afac_agent.domain.models.datasets import ClassificationDataset, RecommendationDataset
from afac_agent.domain.ports.experiment_store import ExperimentStore
from afac_agent.domain.ports.submission_writer import SubmissionWriter
from afac_agent.infrastructure.algorithms.factory import (
from afac_agent.infrastructure.algorithms.factory import (
    build_classification_algorithm,
    build_recommendation_algorithm,
)
from afac_agent.infrastructure.config.schema import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExperimentAgent:
    """面向 A1/A2 的串行实验搜索与提交生成 Agent。"""

    config: AppConfig
    dataset_loader: DatasetLoader
    submission_writer: SubmissionWriter
    experiment_store: ExperimentStore

    def run_all(self) -> str:
        """运行 A1/A2 实验并生成 prediction.zip。"""
        self._seed_everything(self.config.run.seed)

        cls_dataset = self.dataset_loader.load_classification()
        rec_dataset = self.dataset_loader.load_recommendation()

        best_cls_candidate = self._search_classification(cls_dataset)
        best_rec_candidate = self._search_recommendation(rec_dataset)

        a1_rows = self._train_full_classification(cls_dataset, best_cls_candidate)
        a2_rows = self._train_full_recommendation(rec_dataset, best_rec_candidate)

        a1_path = self.submission_writer.write_a1(a1_rows)
        a2_path = self.submission_writer.write_a2(a2_rows)
        zip_path = self.submission_writer.write_zip(a1_path, a2_path)

        self.experiment_store.append(
            {
                "time": _utc_now(),
                "event": "final_submission_generated",
                "a1_path": a1_path,
                "a2_path": a2_path,
                "zip_path": zip_path,
                "best_classification": best_cls_candidate,
                "best_recommendation": best_rec_candidate,
            }
        )
        return zip_path

    def _search_classification(self, dataset: ClassificationDataset) -> dict[str, Any]:
        """对分类任务做串行实验搜索。"""
        split = stratified_split_indices(
            indices=dataset.train_idx.tolist(),
            labels=np.asarray(dataset.labels),
            val_ratio=self.config.data.classification.val_ratio,
            seed=self.config.run.seed,
        )

        best: dict[str, Any] | None = None
        best_score = float("-inf")
        patience_left = int(self.config.run.experiment.patience)

        candidates = list(self.config.search_space.classification_candidates)[: self.config.run.experiment.max_trials]
        for trial_id, cand in enumerate(candidates, start=1):
            algo = build_classification_algorithm(cand)
            result = algo.run(dataset, train_idx=split.train, val_idx=split.val)
            score = float(result.val_accuracy or 0.0)

            record = {
                "time": _utc_now(),
                "task": "classification",
                "trial_id": trial_id,
                "candidate": cand,
                "val_accuracy": score,
            }
            self.experiment_store.append(record)

            improved = score > best_score
            if improved:
                best_score = score
                best = cand
                patience_left = int(self.config.run.experiment.patience)
                logger.info("Best classification updated: val_accuracy=%.6f candidate=%s", score, cand)
            else:
                patience_left -= 1
                logger.info("Classification no improvement, patience_left=%d", patience_left)

            if patience_left <= 0:
                logger.info("Classification early stop triggered")
                break

        if best is None:
            raise ValueError("no classification candidate produced a result")
        return best

    def _search_recommendation(self, dataset: RecommendationDataset) -> dict[str, Any]:
        """对推荐任务做串行实验搜索。"""
        split = random_split_row_indices(
            row_count=len(dataset.train),
            val_ratio=self.config.data.recommendation.val_ratio,
            seed=self.config.run.seed,
        )

        best: dict[str, Any] | None = None
        best_score = float("-inf")
        patience_left = int(self.config.run.experiment.patience)

        candidates = list(self.config.search_space.recommendation_candidates)[: self.config.run.experiment.max_trials]
        for trial_id, cand in enumerate(candidates, start=1):
            algo = build_recommendation_algorithm(cand)
            result = algo.run(dataset, train_row_idx=split.train, val_row_idx=split.val)
            score = float(result.val_ndcg_at_10 or 0.0)

            record = {
                "time": _utc_now(),
                "task": "recommendation",
                "trial_id": trial_id,
                "candidate": cand,
                "val_ndcg_at_10": score,
            }
            self.experiment_store.append(record)

            improved = score > best_score
            if improved:
                best_score = score
                best = cand
                patience_left = int(self.config.run.experiment.patience)
                logger.info("Best recommendation updated: val_ndcg@10=%.6f candidate=%s", score, cand)
            else:
                patience_left -= 1
                logger.info("Recommendation no improvement, patience_left=%d", patience_left)

            if patience_left <= 0:
                logger.info("Recommendation early stop triggered")
                break

        if best is None:
            raise ValueError("no recommendation candidate produced a result")
        return best

    def _train_full_classification(self, dataset: ClassificationDataset, candidate: dict[str, Any]):
        """使用最佳配置在全量训练集上训练并预测测试集。"""
        algo = build_classification_algorithm(candidate)
        result = algo.run(dataset, train_idx=dataset.train_idx.tolist(), val_idx=[])
        return result.test_predictions

    def _train_full_recommendation(self, dataset: RecommendationDataset, candidate: dict[str, Any]):
        """使用最佳配置在全量训练集上训练并预测测试集。"""
        algo = build_recommendation_algorithm(candidate)
        full_train_idx = list(range(len(dataset.train)))
        result = algo.run(dataset, train_row_idx=full_train_idx, val_row_idx=[])
        return result.test_predictions

    def _seed_everything(self, seed: int) -> None:
        """设置随机种子。"""
        random.seed(seed)
        np.random.seed(seed)


def _utc_now() -> str:
    """返回 UTC 时间戳字符串。"""
    return datetime.now(timezone.utc).isoformat()
