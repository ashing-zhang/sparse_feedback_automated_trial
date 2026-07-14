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
from typing import Any

import numpy as np

from afac_agent.domain.ports.dataset_loader import DatasetLoader
from afac_agent.domain.models.datasets import ClassificationDataset, RecommendationDataset
from afac_agent.domain.ports.experiment_store import ExperimentStore
from afac_agent.domain.ports.submission_writer import SubmissionWriter
from afac_agent.infrastructure.algorithms.factory import (
    build_classification_algorithm,
    build_recommendation_algorithm,
)
from afac_agent.infrastructure.config.schema import AppConfig
from afac_agent.application.services.experiment_controller import ExperimentController

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExperimentAgent:
    """面向 A1/A2 的串行实验搜索与提交生成 Agent。"""

    config: AppConfig
    dataset_loader: DatasetLoader
    submission_writer: SubmissionWriter
    experiment_store: ExperimentStore
    controller: ExperimentController

    def run_all(self) -> str:
        """运行 A1/A2 实验并生成 prediction.zip。"""
        self._seed_everything(self.config.run.seed)

        cls_dataset = self.dataset_loader.load_classification()
        rec_dataset = self.dataset_loader.load_recommendation()

        best_cls_candidate = self.controller.run_classification(cls_dataset)
        best_rec_candidate = self.controller.run_recommendation(rec_dataset)

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
