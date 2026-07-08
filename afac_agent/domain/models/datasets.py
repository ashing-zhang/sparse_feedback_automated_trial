"""
运行指南：
- 本文件为领域层数据结构定义，不直接运行。
- 由基础设施层负责从 data/ 目录读取文件并构造这些数据结构，再交由应用层编排。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix


@dataclass(frozen=True, slots=True)
class ClassificationDataset:
    """产品分类任务数据（图节点分类）。"""

    adjacency: "csr_matrix"
    attributes: "csr_matrix"
    labels: "np.ndarray"
    train_idx: "np.ndarray"
    test_idx: "np.ndarray"


@dataclass(frozen=True, slots=True)
class RecommendationDataset:
    """产品推荐任务数据（Top10 排序）。"""

    train: "pd.DataFrame"
    test: "pd.DataFrame"
    users: "pd.DataFrame"
    items: "pd.DataFrame"

