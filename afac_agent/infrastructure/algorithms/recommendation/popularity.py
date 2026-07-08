"""
运行指南：
- 本模块实现推荐任务的 Popularity 基线算法，不直接运行。
- 由 Experiment Agent 按配置创建该算法并执行 run()。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import logging

import pandas as pd

from afac_agent.application.services.metrics import ndcg_at_k_single_target
from afac_agent.domain.models.datasets import RecommendationDataset
from afac_agent.domain.models.predictions import A2PredictionRow
from afac_agent.domain.models.runs import RecommendationRunResult
from afac_agent.infrastructure.algorithms.recommendation.utils import dedupe_keep_order, parse_item_seq

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PopularityAlgorithm:
    """Popularity（按全局热门度推荐，过滤用户历史）。"""

    top_k: int = 10

    def run(
        self,
        dataset: RecommendationDataset,
        train_row_idx: list[int],
        val_row_idx: list[int],
    ) -> RecommendationRunResult:
        """训练并评估，同时对测试集生成预测。"""
        train_df = dataset.train
        items_all = dataset.items["iid"].astype(str).tolist()

        pop = _fit_popularity(train_df, train_row_idx)
        pop_rank = _rank_items(items_all, pop)

        val_ndcg: float | None = None
        if val_row_idx:
            targets, preds = _predict_for_rows(train_df, val_row_idx, pop_rank, self.top_k)
            val_ndcg = ndcg_at_k_single_target(targets, preds, k=self.top_k)
            logger.info("Popularity val_ndcg@%d=%.6f", self.top_k, val_ndcg)

        test_predictions = _predict_for_test(dataset.test, pop_rank, self.top_k)
        return RecommendationRunResult(val_ndcg_at_10=val_ndcg, test_predictions=test_predictions)


def _fit_popularity(train_df: pd.DataFrame, row_idx: list[int]) -> Counter[str]:
    """从 train 子集拟合 item 热门度。"""
    if not row_idx:
        row_idx = list(range(len(train_df)))
    targets = train_df.iloc[row_idx]["target_iid"].astype(str).tolist()
    return Counter(targets)


def _rank_items(items_all: list[str], pop: Counter[str]) -> list[str]:
    """对 item 按热门度排序（稳定排序：再按 iid）。"""
    return sorted(items_all, key=lambda x: (-int(pop.get(x, 0)), x))


def _recommend_from_rank(pop_rank: list[str], history: list[str], top_k: int) -> list[str]:
    """从全局排序中过滤历史并取 top_k。"""
    hist = set(history)
    out: list[str] = []
    for iid in pop_rank:
        if iid in hist:
            continue
        out.append(iid)
        if len(out) >= top_k:
            break
    return out


def _predict_for_rows(
    train_df: pd.DataFrame,
    row_idx: list[int],
    pop_rank: list[str],
    top_k: int,
) -> tuple[list[str], list[list[str]]]:
    """对 train 的指定行做预测（用于验证）。"""
    sub = train_df.iloc[row_idx]
    targets = sub["target_iid"].astype(str).tolist()
    preds: list[list[str]] = []
    for seq in sub["item_seq_dedup"].astype(str).tolist():
        history = dedupe_keep_order(parse_item_seq(seq))
        preds.append(_recommend_from_rank(pop_rank, history, top_k))
    return targets, preds


def _predict_for_test(test_df: pd.DataFrame, pop_rank: list[str], top_k: int) -> list[A2PredictionRow]:
    """对官方 test.csv 生成预测提交行。"""
    rows: list[A2PredictionRow] = []
    for uid, seq in zip(test_df["uid"].astype(str), test_df["item_seq_dedup"].astype(str), strict=False):
        history = dedupe_keep_order(parse_item_seq(seq))
        pred = _recommend_from_rank(pop_rank, history, top_k)
        rows.append(A2PredictionRow(uid=str(uid), prediction=pred))
    return rows

