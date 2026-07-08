"""
运行指南：
- 本模块提供训练/验证划分工具函数，不直接运行。
- 分类任务：基于 train_idx + labels 做分层划分
- 推荐任务：对 train.csv 行索引做随机划分
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """通用划分索引结果。"""

    train: list[int]
    val: list[int]


def stratified_split_indices(
    indices: Iterable[int],
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> SplitIndices:
    """对给定索引集合按 labels 分层划分 train/val。"""
    idx_list = list(int(i) for i in indices)
    y = np.asarray([int(labels[i]) for i in idx_list], dtype=int)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_pos, val_pos = next(splitter.split(idx_list, y))
    train_idx = [idx_list[i] for i in train_pos.tolist()]
    val_idx = [idx_list[i] for i in val_pos.tolist()]
    return SplitIndices(train=train_idx, val=val_idx)


def random_split_row_indices(
    row_count: int,
    val_ratio: float,
    seed: int,
) -> SplitIndices:
    """对 [0..row_count) 行索引做随机划分。"""
    if row_count <= 0:
        return SplitIndices(train=[], val=[])
    all_idx = list(range(row_count))
    rng = random.Random(seed)
    rng.shuffle(all_idx)
    val_size = int(round(row_count * val_ratio))
    val_idx = all_idx[:val_size]
    train_idx = all_idx[val_size:]
    return SplitIndices(train=train_idx, val=val_idx)

