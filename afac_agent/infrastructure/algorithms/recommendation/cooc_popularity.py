"""
运行指南：
- 本模块实现推荐任务的 Co-Occurrence + Popularity 混合基线算法，不直接运行。
- 由 Experiment Agent 按配置创建该算法并执行 run()。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
import logging

import pandas as pd

from afac_agent.application.services.metrics import ndcg_at_k_single_target
from afac_agent.domain.models.datasets import RecommendationDataset
from afac_agent.domain.models.predictions import A2PredictionRow
from afac_agent.domain.models.runs import RecommendationRunResult
from afac_agent.infrastructure.algorithms.recommendation.utils import dedupe_keep_order, parse_item_seq

logger = logging.getLogger(__name__)


CoocMap = dict[str, Counter[str]]


@dataclass(frozen=True, slots=True)
class CoocPopularityAlgorithm:
    """按 co-occurrence 召回 + 热门度兜底的混合排序。"""

    cooc_window: int
    cooc_weight: float
    popularity_weight: float
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
        cooc = _fit_cooc(train_df, train_row_idx, window=self.cooc_window)

        val_ndcg: float | None = None
        if val_row_idx:
            targets, preds = _predict_for_rows(
                train_df,
                val_row_idx,
                pop,
                pop_rank,
                cooc,
                self.top_k,
                self.cooc_weight,
                self.popularity_weight,
            )
            val_ndcg = ndcg_at_k_single_target(targets, preds, k=self.top_k)
            logger.info("CoocPopularity val_ndcg@%d=%.6f", self.top_k, val_ndcg)

        test_predictions = _predict_for_test(
            dataset.test,
            pop,
            pop_rank,
            cooc,
            self.top_k,
            self.cooc_weight,
            self.popularity_weight,
        )
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


def _fit_cooc(train_df: pd.DataFrame, row_idx: list[int], window: int) -> CoocMap:
    """拟合 item co-occurrence（基于 item_seq_dedup）。"""
    if not row_idx:
        row_idx = list(range(len(train_df)))

    cooc: dict[str, Counter[str]] = defaultdict(Counter)
    for seq_raw in train_df.iloc[row_idx]["item_seq_dedup"].astype(str).tolist():
        seq = dedupe_keep_order(parse_item_seq(seq_raw))
        if window > 0 and len(seq) > window:
            seq = seq[-window:]
        for i, a in enumerate(seq):
            for b in seq[i + 1 :]:
                if a == b:
                    continue
                cooc[a][b] += 1
                cooc[b][a] += 1
    return dict(cooc)


def _score_candidates(
    history: list[str],
    pop: Counter[str],
    cooc: CoocMap,
    cooc_weight: float,
    popularity_weight: float,
) -> Counter[str]:
    """基于 history 生成候选 item 的分数。"""
    scores: Counter[str] = Counter()
    for h in history:
        for item, c in cooc.get(h, {}).items():
            scores[item] += float(cooc_weight) * float(c)
    if popularity_weight != 0.0:
        for item, c in pop.items():
            scores[item] += float(popularity_weight) * float(c)
    return scores


def _recommend(
    history: list[str],
    pop_rank: list[str],
    scores: Counter[str],
    top_k: int,
) -> list[str]:
    """按 scores 排序并用 pop_rank 兜底，过滤历史并截断 top_k。"""
    hist = set(history)
    out: list[str] = []

    for item, _ in scores.most_common():
        if item in hist:
            continue
        out.append(item)
        if len(out) >= top_k:
            return out

    for item in pop_rank:
        if item in hist or item in out:
            continue
        out.append(item)
        if len(out) >= top_k:
            return out

    return out


def _predict_for_rows(
    train_df: pd.DataFrame,
    row_idx: list[int],
    pop: Counter[str],
    pop_rank: list[str],
    cooc: CoocMap,
    top_k: int,
    cooc_weight: float,
    popularity_weight: float,
) -> tuple[list[str], list[list[str]]]:
    """对 train 的指定行做预测（用于验证）。"""
    sub = train_df.iloc[row_idx]
    targets = sub["target_iid"].astype(str).tolist()
    preds: list[list[str]] = []
    for seq in sub["item_seq_dedup"].astype(str).tolist():
        history = dedupe_keep_order(parse_item_seq(seq))
        scores = _score_candidates(history, pop, cooc, cooc_weight, popularity_weight)
        preds.append(_recommend(history, pop_rank, scores, top_k))
    return targets, preds


def _predict_for_test(
    test_df: pd.DataFrame,
    pop: Counter[str],
    pop_rank: list[str],
    cooc: CoocMap,
    top_k: int,
    cooc_weight: float,
    popularity_weight: float,
) -> list[A2PredictionRow]:
    """对官方 test.csv 生成预测提交行。"""
    rows: list[A2PredictionRow] = []
    for uid, seq in zip(test_df["uid"].astype(str), test_df["item_seq_dedup"].astype(str), strict=False):
        history = dedupe_keep_order(parse_item_seq(seq))
        scores = _score_candidates(history, pop, cooc, cooc_weight, popularity_weight)
        pred = _recommend(history, pop_rank, scores, top_k)
        rows.append(A2PredictionRow(uid=str(uid), prediction=pred))
    return rows

