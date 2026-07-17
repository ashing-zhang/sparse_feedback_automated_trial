"""
运行指南：
- 本模块实现推荐任务的 LightGCN 算法，不直接运行。
- 由 Experiment Agent 按配置创建该算法并执行 run()。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.sparse import csr_matrix

from afac_agent.application.services.metrics import ndcg_at_k_single_target
from afac_agent.domain.models.datasets import RecommendationDataset
from afac_agent.domain.models.predictions import A2PredictionRow
from afac_agent.domain.models.runs import RecommendationRunResult
from afac_agent.infrastructure.algorithms.recommendation.utils import dedupe_keep_order, parse_item_seq

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LightGCNAlgorithm:
    """LightGCN（基于图卷积的推荐算法）。"""

    hidden_dim: int
    num_layers: int
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
        adj_norm = _normalize_adjacency(adj_matrix)

        model = _LightGCNModel(
            num_users=num_users,
            num_items=num_items,
            hidden_dim=int(self.hidden_dim),
            num_layers=int(self.num_layers),
            adj_norm=adj_norm,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed))

        train_df = dataset.train

        logger.info(
            "LightGCN initialized hidden_dim=%d num_layers=%d lr=%.6f weight_decay=%.6f epochs=%d patience=%d",
            self.hidden_dim,
            self.num_layers,
            self.learning_rate,
            self.weight_decay,
            self.epochs,
            self.patience,
        )

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

                jidx = np.random.randint(0, num_items)
                while jidx in pos_items:
                    jidx = np.random.randint(0, num_items)

                optimizer.zero_grad()
                loss = model.bpr_loss(uidx, iidx, jidx)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                count += 1

            if count > 0:
                logger.info("LightGCN epoch=%d loss=%.6f", epoch + 1, total_loss / count)

            if val_row_idx:
                model.eval()
                user_emb, item_emb = model.get_embeddings()
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
                    logger.info("LightGCN early stopping at epoch=%d, best_val_ndcg@%d=%.6f", epoch + 1, self.top_k, best_val_ndcg)
                    break

        model.eval()
        user_emb, item_emb = model.get_embeddings()

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
            logger.info("LightGCN val_ndcg@%d=%.6f", self.top_k, val_ndcg)

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


class _LightGCNModel(nn.Module):
    """LightGCN 模型。"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        hidden_dim: int,
        num_layers: int,
        adj_norm: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("adj_norm", adj_norm)
        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.item_emb = nn.Embedding(num_items, hidden_dim)
        self.num_layers = num_layers

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """执行多层传播，返回 user/item 嵌入。"""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]

        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(self.adj_norm, all_emb)
            embs.append(all_emb)

        final_emb = torch.stack(embs, dim=0).mean(dim=0)
        num_users = self.user_emb.weight.shape[0]
        return final_emb[:num_users], final_emb[num_users:]

    def bpr_loss(self, u: int, i: int, j: int) -> torch.Tensor:
        """Bayesian Personalized Ranking loss。"""
        user_emb, item_emb = self.forward()
        pos_score = torch.dot(user_emb[u], item_emb[i])
        neg_score = torch.dot(user_emb[u], item_emb[j])
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10)
        return loss

    def get_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """获取当前 user/item 嵌入。"""
        with torch.no_grad():
            return self.forward()


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


def _normalize_adjacency(adj: csr_matrix) -> torch.Tensor:
    """对称归一化邻接矩阵。"""
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(np.maximum(deg, 1), -0.5)
    deg_inv_sqrt = csr_matrix(np.diag(deg_inv_sqrt))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt

    coo = adj_norm.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, adj_norm.shape)


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