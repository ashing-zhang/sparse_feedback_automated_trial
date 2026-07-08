"""
运行指南：
- 本模块从本地 data/ 目录加载 A 榜数据，不直接运行。
- 由入口模块组装为 DatasetLoader 端口实现供应用层调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from afac_agent.domain.models.datasets import ClassificationDataset, RecommendationDataset
from afac_agent.infrastructure.config.schema import DataConfig

logger = logging.getLogger(__name__)


def _build_csr(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
) -> csr_matrix:
    """从 CSR 三元组构造 csr_matrix。"""
    return csr_matrix((data, indices, indptr), shape=shape)


@dataclass(frozen=True, slots=True)
class FileDatasetLoader:
    """从文件系统加载数据集。"""

    data_config: DataConfig

    def load_classification(self) -> ClassificationDataset:
        """加载产品分类任务数据（npz）。"""
        path = self.data_config.classification.npz_path
        logger.info("Loading classification dataset: %s", path)

        with np.load(path, allow_pickle=False) as npz:
            adj = _build_csr(
                data=npz["adj_data"],
                indices=npz["adj_indices"],
                indptr=npz["adj_indptr"],
                shape=tuple(int(x) for x in npz["adj_shape"]),
            )
            attr = _build_csr(
                data=npz["attr_data"],
                indices=npz["attr_indices"],
                indptr=npz["attr_indptr"],
                shape=tuple(int(x) for x in npz["attr_shape"]),
            )
            labels = npz["labels"]
            train_idx = npz["train_idx"]
            test_idx = npz["test_idx"]

        return ClassificationDataset(
            adjacency=adj,
            attributes=attr,
            labels=labels,
            train_idx=train_idx,
            test_idx=test_idx,
        )

    def load_recommendation(self) -> RecommendationDataset:
        """加载产品推荐任务数据（csv）。"""
        cfg = self.data_config.recommendation

        logger.info("Loading recommendation train: %s", cfg.train_csv_path)
        train_df = pd.read_csv(cfg.train_csv_path)
        logger.info("Loading recommendation test: %s", cfg.test_csv_path)
        test_df = pd.read_csv(cfg.test_csv_path)
        logger.info("Loading recommendation users: %s", cfg.user_csv_path)
        user_df = pd.read_csv(cfg.user_csv_path)
        logger.info("Loading recommendation items: %s", cfg.item_csv_path)
        item_df = pd.read_csv(cfg.item_csv_path)

        _validate_recommendation_frames(train_df, test_df, user_df, item_df)

        return RecommendationDataset(train=train_df, test=test_df, users=user_df, items=item_df)


def _validate_recommendation_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
) -> None:
    """校验推荐任务数据表字段。"""
    required_train = {"uid", "target_iid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"}
    required_test = {"uid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"}
    required_user = {"uid"}
    required_item = {"iid"}

    if not required_train.issubset(set(train_df.columns)):
        raise ValueError(f"train.csv missing columns: {sorted(required_train - set(train_df.columns))}")
    if not required_test.issubset(set(test_df.columns)):
        raise ValueError(f"test.csv missing columns: {sorted(required_test - set(test_df.columns))}")
    if not required_user.issubset(set(user_df.columns)):
        raise ValueError(f"user.csv missing columns: {sorted(required_user - set(user_df.columns))}")
    if not required_item.issubset(set(item_df.columns)):
        raise ValueError(f"item.csv missing columns: {sorted(required_item - set(item_df.columns))}")

