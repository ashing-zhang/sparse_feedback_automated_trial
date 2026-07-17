"""
运行指南：
- 本模块实现推荐任务的 Graph Transformer 算法，不直接运行。
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
from scipy.sparse import csr_matrix

from afac_agent.application.services.metrics import ndcg_at_k_single_target
from afac_agent.domain.models.datasets import RecommendationDataset
from afac_agent.domain.models.predictions import A2PredictionRow
from afac_agent.domain.models.runs import RecommendationRunResult
from afac_agent.infrastructure.algorithms.recommendation.utils import dedupe_keep_order, parse_item_seq

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GraphTransformerAlgorithm:
    """Graph Transformer（基于 Transformer 的图推荐算法）。"""

    hidden_dim: int
    num_layers: int
    num_heads: int
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
        users_all = dataset.users["uid"].astype(str).tolist()
        items_all = dataset.items["iid"].astype(str).tolist()

        user_to_idx = {user: idx for idx, user in enumerate(users_all)}
        item_to_idx = {item: idx for idx, item in enumerate(items_all)}

        num_users = len(user_to_idx)
        num_items = len(item_to_idx)

        adj_matrix = _build_adjacency(
            dataset.train,
            train_row_idx,
            user_to_idx,
            item_to_idx,
            num_users,
            num_items,
        )

        model = _GraphTransformerModel(
            num_users=num_users,
            num_items=num_items,
            hidden_dim=int(self.hidden_dim),
            num_layers=int(self.num_layers),
            num_heads=int(self.num_heads),
            dropout=float(self.dropout),
            adj_matrix=adj_matrix,
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
            "GraphTransformer initialized hidden_dim=%d num_layers=%d num_heads=%d dropout=%.3f lr=%.6f weight_decay=%.6f epochs=%d patience=%d",
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
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

            for uid_raw, seq_raw, target in zip(
                train_df.iloc[train_row_idx]["uid"].astype(str),
                train_df.iloc[train_row_idx]["item_seq_dedup"].astype(str),
                train_df.iloc[train_row_idx]["target_iid"].astype(str),
                strict=False,
            ):
                uidx = user_to_idx.get(uid_raw)
                if uidx is None:
                    continue
                iidx = item_to_idx.get(target)
                if iidx is None:
                    continue

                history = dedupe_keep_order(parse_item_seq(seq_raw))
                pos_items = [item_to_idx.get(i) for i in history if item_to_idx.get(i) is not None]
                if iidx in pos_items:
                    pos_items.remove(iidx)

                if not pos_items:
                    continue

                optimizer.zero_grad()
                user_emb, item_emb = model()
                logits = torch.mm(user_emb[uidx].unsqueeze(0), item_emb.T)
                loss = criterion(logits, torch.tensor([iidx], dtype=torch.long))
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                count += 1

            if count > 0:
                logger.info("GraphTransformer epoch=%d loss=%.6f", epoch + 1, total_loss / count)

            if val_row_idx:
                model.eval()
                user_emb, item_emb = model()
                targets, preds = _predict_for_rows(
                    train_df,
                    val_row_idx,
                    user_to_idx,
                    item_to_idx,
                    items_all,
                    user_emb,
                    item_emb,
                    self.top_k,
                )
                current_val_ndcg = ndcg_at_k_single_target(targets, preds, k=self.top_k)

                if current_val_ndcg > best_val_ndcg:
                    best_val_ndcg = current_val_ndcg
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info("GraphTransformer early stopping at epoch=%d, best_val_ndcg@%d=%.6f", epoch + 1, self.top_k, best_val_ndcg)
                    break

        model.eval()

        user_emb, item_emb = model()

        val_ndcg: float | None = None
        if val_row_idx:
            targets, preds = _predict_for_rows(
                train_df,
                val_row_idx,
                user_to_idx,
                item_to_idx,
                items_all,
                user_emb,
                item_emb,
                self.top_k,
            )
            val_ndcg = ndcg_at_k_single_target(targets, preds, k=self.top_k)
            logger.info("GraphTransformer val_ndcg@%d=%.6f", self.top_k, val_ndcg)

        test_predictions = _predict_for_test(
            dataset.test,
            user_to_idx,
            item_to_idx,
            items_all,
            user_emb,
            item_emb,
            self.top_k,
        )
        return RecommendationRunResult(val_ndcg_at_10=val_ndcg, test_predictions=test_predictions)


class _GraphTransformerModel(nn.Module):
    """Graph Transformer 推荐模型。"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        adj_matrix: csr_matrix,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.item_emb = nn.Embedding(num_items, hidden_dim)
        self.num_users = num_users

        self.layers = nn.ModuleList([
            _GraphTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        adj_coo = adj_matrix.tocoo()
        self.register_buffer(
            "adj_indices",
            torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long),
        )
        self.register_buffer(
            "adj_values",
            torch.tensor(adj_coo.data, dtype=torch.float32),
        )
        self.adj_shape = adj_matrix.shape

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """前向计算，返回 user/item 嵌入。"""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        adj = torch.sparse_coo_tensor(
            self.adj_indices,
            self.adj_values,
            self.adj_shape,
            device=all_emb.device,
        )

        for layer in self.layers:
            all_emb = layer(all_emb, adj)

        return all_emb[: self.num_users], all_emb[self.num_users :]


class _GraphTransformerLayer(nn.Module):
    """Graph Transformer 层。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """执行一次 Transformer 层计算。"""
        x_neighbor = torch.sparse.mm(adj, x)
        x_neighbor = x_neighbor.unsqueeze(0)
        x_self = x.unsqueeze(0)

        attn_out, _ = self.attention(x_self, x_neighbor, x_neighbor)
        x = self.norm1(x + attn_out.squeeze(0))

        mlp_out = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_out))

        return x


def _build_adjacency(
    train_df: pd.DataFrame,
    row_idx: list[int],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    num_users: int,
    num_items: int,
) -> csr_matrix:
    """构建 user-item 交互邻接矩阵。"""
    if not row_idx:
        row_idx = list(range(len(train_df)))

    rows: list[int] = []
    cols: list[int] = []

    for uid_raw, seq_raw, target in zip(
        train_df.iloc[row_idx]["uid"].astype(str),
        train_df.iloc[row_idx]["item_seq_dedup"].astype(str),
        train_df.iloc[row_idx]["target_iid"].astype(str),
        strict=False,
    ):
        uidx = user_to_idx.get(uid_raw)
        if uidx is None:
            continue

        history = dedupe_keep_order(parse_item_seq(seq_raw))
        for item in history:
            iidx = item_to_idx.get(item)
            if iidx is not None:
                rows.append(uidx)
                cols.append(num_users + iidx)
                rows.append(num_users + iidx)
                cols.append(uidx)

        iidx = item_to_idx.get(target)
        if iidx is not None:
            rows.append(uidx)
            cols.append(num_users + iidx)
            rows.append(num_users + iidx)
            cols.append(uidx)

    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix(
        (data, (rows, cols)),
        shape=(num_users + num_items, num_users + num_items),
    )
    return adj


def _predict_for_rows(
    df: pd.DataFrame,
    row_idx: list[int],
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    items_all: list[str],
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    top_k: int,
) -> tuple[list[str], list[list[str]]]:
    """对指定行做预测（用于验证）。"""
    sub = df.iloc[row_idx]
    targets = sub["target_iid"].astype(str).tolist()
    preds: list[list[str]] = []

    with torch.no_grad():
        for uid_raw, seq_raw in zip(sub["uid"].astype(str), sub["item_seq_dedup"].astype(str), strict=False):
            uidx = user_to_idx.get(uid_raw)
            if uidx is None:
                preds.append(items_all[:top_k])
                continue

            history = dedupe_keep_order(parse_item_seq(seq_raw))
            hist_set = set(history)

            scores = user_emb[uidx] @ item_emb.T
            _, indices = torch.topk(scores, top_k + len(hist_set))

            recommendation = []
            for idx in indices.tolist():
                item = items_all[idx]
                if item not in hist_set and item not in recommendation:
                    recommendation.append(item)
                    if len(recommendation) >= top_k:
                        break

            preds.append(recommendation)

    return targets, preds


def _predict_for_test(
    test_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    items_all: list[str],
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    top_k: int,
) -> list[A2PredictionRow]:
    """对官方 test.csv 生成预测提交行。"""
    rows: list[A2PredictionRow] = []

    with torch.no_grad():
        for uid_raw, seq_raw in zip(test_df["uid"].astype(str), test_df["item_seq_dedup"].astype(str), strict=False):
            uidx = user_to_idx.get(uid_raw)
            if uidx is None:
                recommendation = items_all[:top_k]
            else:
                history = dedupe_keep_order(parse_item_seq(seq_raw))
                hist_set = set(history)

                scores = user_emb[uidx] @ item_emb.T
                _, indices = torch.topk(scores, top_k + len(hist_set))

                recommendation = []
                for idx in indices.tolist():
                    item = items_all[idx]
                    if item not in hist_set and item not in recommendation:
                        recommendation.append(item)
                        if len(recommendation) >= top_k:
                            break

            rows.append(A2PredictionRow(uid=str(uid_raw), prediction=recommendation))

    return rows