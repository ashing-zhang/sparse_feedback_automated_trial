"""
运行指南：
- 本模块实现推荐任务的 GRU4Rec 序列推荐算法，不直接运行。
- 由 Experiment Agent 按配置创建该算法并执行 run()。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from afac_agent.application.services.metrics import ndcg_at_k_single_target
from afac_agent.domain.models.datasets import RecommendationDataset
from afac_agent.domain.models.predictions import A2PredictionRow
from afac_agent.domain.models.runs import RecommendationRunResult
from afac_agent.infrastructure.algorithms.recommendation.utils import dedupe_keep_order, parse_item_seq

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GRU4RecAlgorithm:
    """GRU4Rec（基于 GRU 的序列推荐）。"""

    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    seed: int = 42
    top_k: int = 10
    patience: int = 5

    def run(
        self,
        dataset: RecommendationDataset,
        train_row_idx: list[int],
        val_row_idx: list[int],
    ) -> RecommendationRunResult:
        """训练并评估，同时对测试集生成预测。"""
        items_all = dataset.items["iid"].astype(str).tolist()
        item_to_idx = {item: idx + 1 for idx, item in enumerate(items_all)}
        vocab_size = len(item_to_idx) + 1

        model = _GRU4RecModel(
            vocab_size=vocab_size,
            hidden_dim=int(self.hidden_dim),
            dropout=float(self.dropout),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        criterion = nn.CrossEntropyLoss()

        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed))

        logger.info(
            "GRU4Rec initialized hidden_dim=%d dropout=%.3f lr=%.6f weight_decay=%.6f epochs=%d patience=%d",
            self.hidden_dim,
            self.dropout,
            self.learning_rate,
            self.weight_decay,
            self.epochs,
            self.patience,
        )

        train_df = dataset.train
        best_val_ndcg = 0.0
        patience_counter = 0

        for epoch in range(int(self.epochs)):
            model.train()
            total_loss = 0.0
            count = 0

            for seq_raw, target in zip(
                train_df.iloc[train_row_idx]["item_seq_dedup"].astype(str),
                train_df.iloc[train_row_idx]["target_iid"].astype(str),
                strict=False,
            ):
                history = dedupe_keep_order(parse_item_seq(seq_raw))
                if len(history) < 1:
                    continue

                seq_ids = torch.tensor([item_to_idx.get(i, 0) for i in history], dtype=torch.long).unsqueeze(0)
                target_idx = item_to_idx.get(target, 0)
                if target_idx == 0:
                    continue

                optimizer.zero_grad()
                logits = model(seq_ids)
                loss = criterion(logits, torch.tensor([target_idx], dtype=torch.long))
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                count += 1

            if count > 0:
                logger.info("GRU4Rec epoch=%d loss=%.6f", epoch + 1, total_loss / count)

            if val_row_idx:
                model.eval()
                targets, preds = _predict_for_rows(
                    train_df,
                    val_row_idx,
                    item_to_idx,
                    items_all,
                    model,
                    self.top_k,
                )
                current_val_ndcg = ndcg_at_k_single_target(targets, preds, k=self.top_k)

                if current_val_ndcg > best_val_ndcg:
                    best_val_ndcg = current_val_ndcg
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info("GRU4Rec early stopping at epoch=%d, best_val_ndcg@%d=%.6f", epoch + 1, self.top_k, best_val_ndcg)
                    break

        model.eval()

        val_ndcg: float | None = None
        if val_row_idx:
            targets, preds = _predict_for_rows(
                train_df,
                val_row_idx,
                item_to_idx,
                items_all,
                model,
                self.top_k,
            )
            val_ndcg = ndcg_at_k_single_target(targets, preds, k=self.top_k)
            logger.info("GRU4Rec val_ndcg@%d=%.6f", self.top_k, val_ndcg)

        test_predictions = _predict_for_test(
            dataset.test,
            item_to_idx,
            items_all,
            model,
            self.top_k,
        )
        return RecommendationRunResult(val_ndcg_at_10=val_ndcg, test_predictions=test_predictions)


class _GRU4RecModel(nn.Module):
    """GRU4Rec 序列推荐模型。"""

    def __init__(self, vocab_size: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = dropout

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """前向计算，返回最后一个位置的 logits。"""
        emb = self.embedding(seq)
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        _, hidden = self.gru(emb)
        logits = self.fc(hidden.squeeze(0))
        return logits


def _predict_for_rows(
    train_df: pd.DataFrame,
    row_idx: list[int],
    item_to_idx: dict[str, int],
    items_all: list[str],
    model: nn.Module,
    top_k: int,
) -> tuple[list[str], list[list[str]]]:
    """对 train 的指定行做预测（用于验证）。"""
    sub = train_df.iloc[row_idx]
    targets = sub["target_iid"].astype(str).tolist()
    preds: list[list[str]] = []

    with torch.no_grad():
        for seq_raw in sub["item_seq_dedup"].astype(str):
            history = dedupe_keep_order(parse_item_seq(seq_raw))
            if len(history) < 1:
                preds.append(items_all[:top_k])
                continue

            seq_ids = torch.tensor([item_to_idx.get(i, 0) for i in history], dtype=torch.long).unsqueeze(0)
            logits = model(seq_ids)
            _, indices = torch.topk(logits, top_k + len(history))

            hist_set = set(history)
            recommendation = []
            for idx in indices[0].tolist():
                if idx == 0:
                    continue
                item = items_all[idx - 1]
                if item not in hist_set and item not in recommendation:
                    recommendation.append(item)
                    if len(recommendation) >= top_k:
                        break

            if len(recommendation) < top_k:
                for item in items_all:
                    if item not in hist_set and item not in recommendation:
                        recommendation.append(item)
                        if len(recommendation) >= top_k:
                            break

            preds.append(recommendation)

    return targets, preds


def _predict_for_test(
    test_df: pd.DataFrame,
    item_to_idx: dict[str, int],
    items_all: list[str],
    model: nn.Module,
    top_k: int,
) -> list[A2PredictionRow]:
    """对官方 test.csv 生成预测提交行。"""
    rows: list[A2PredictionRow] = []

    with torch.no_grad():
        for uid, seq_raw in zip(test_df["uid"].astype(str), test_df["item_seq_dedup"].astype(str), strict=False):
            history = dedupe_keep_order(parse_item_seq(seq_raw))

            if len(history) < 1:
                recommendation = items_all[:top_k]
            else:
                seq_ids = torch.tensor([item_to_idx.get(i, 0) for i in history], dtype=torch.long).unsqueeze(0)
                logits = model(seq_ids)
                _, indices = torch.topk(logits, top_k + len(history))

                hist_set = set(history)
                recommendation = []
                for idx in indices[0].tolist():
                    if idx == 0:
                        continue
                    item = items_all[idx - 1]
                    if item not in hist_set and item not in recommendation:
                        recommendation.append(item)
                        if len(recommendation) >= top_k:
                            break

                if len(recommendation) < top_k:
                    for item in items_all:
                        if item not in hist_set and item not in recommendation:
                            recommendation.append(item)
                            if len(recommendation) >= top_k:
                                break

            rows.append(A2PredictionRow(uid=str(uid), prediction=recommendation))

    return rows