"""
运行指南：
- 本模块提供 GCN / GraphSAGE 共用的图神经网络工具函数，不直接运行。
- 由分类算法模块调用，用于图矩阵预处理、PyTorch 稀疏张量转换与全图训练。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, eye
import torch
from torch import nn

from afac_agent.application.services.metrics import accuracy
from afac_agent.domain.models.predictions import A1PredictionRow
from afac_agent.domain.models.runs import ClassificationRunResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GNNTrainingConfig:
    """GNN 训练配置。"""

    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int = 42


def prepare_gcn_adjacency(adjacency: csr_matrix) -> csr_matrix:
    """构造带自环的对称归一化邻接矩阵。"""
    sym = _symmetrize(adjacency)
    with_self_loop = (sym + eye(sym.shape[0], format="csr")).tocsr()
    degree = np.asarray(with_self_loop.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5, dtype=np.float32)
    norm = diags(inv_sqrt)
    return (norm @ with_self_loop @ norm).tocsr()


def prepare_mean_adjacency(adjacency: csr_matrix) -> csr_matrix:
    """构造行归一化的邻接矩阵，用于邻居均值聚合。"""
    sym = _symmetrize(adjacency)
    degree = np.asarray(sym.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv[nonzero] = 1.0 / degree[nonzero]
    norm = diags(inv)
    return (norm @ sym).tocsr()


def infer_num_classes(labels: np.ndarray, train_idx: np.ndarray) -> int:
    """根据已标注训练节点推断类别数。"""
    labeled = np.asarray(labels[np.asarray(train_idx, dtype=int)], dtype=int)
    if labeled.size == 0:
        raise ValueError("empty labeled nodes")
    return int(np.max(labeled)) + 1


def to_torch_sparse(matrix: csr_matrix, device: torch.device) -> torch.Tensor:
    """将 SciPy CSR 矩阵转为 PyTorch COO 稀疏张量。"""
    coo: coo_matrix = matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
    value_tensor = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(index_tensor, value_tensor, size=coo.shape, device=device).coalesce()


def linear_sparse_or_dense(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """对稀疏或稠密输入执行线性变换。"""
    if inputs.is_sparse:
        return torch.sparse.mm(inputs, weight)
    return inputs @ weight


def select_device() -> torch.device:
    """选择可用计算设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_torch_seed(seed: int) -> None:
    """设置 PyTorch 随机种子。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_node_classifier(
    model: nn.Module,
    *,
    features: csr_matrix,
    labels: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: np.ndarray,
    config: GNNTrainingConfig,
    algorithm_name: str,
) -> ClassificationRunResult:
    """执行全图训练并返回验证指标与测试预测。"""
    if not train_idx:
        raise ValueError("empty train_idx")

    set_torch_seed(config.seed)
    device = select_device()
    logger.info("%s training on device=%s", algorithm_name, device)

    features_tensor = to_torch_sparse(features, device)
    labels_tensor = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long, device=device)
    train_index_tensor = torch.as_tensor(np.asarray(train_idx, dtype=np.int64), dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        logits = model(features_tensor)
        loss = criterion(logits.index_select(0, train_index_tensor), labels_tensor.index_select(0, train_index_tensor))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch == int(config.epochs) or epoch % 50 == 0:
            logger.info("%s epoch=%d loss=%.6f", algorithm_name, epoch, float(loss.item()))

    model.eval()
    with torch.no_grad():
        pred_all = model(features_tensor).argmax(dim=1).cpu().numpy().astype(int)

    val_acc: float | None = None
    if val_idx:
        y_true = [int(labels[i]) for i in val_idx]
        y_pred = [int(pred_all[i]) for i in val_idx]
        val_acc = accuracy(y_true, y_pred)
        logger.info("%s val_accuracy=%.6f", algorithm_name, val_acc)

    test_predictions = [
        A1PredictionRow(test_idx=int(idx), label=int(label))
        for idx, label in zip(test_idx.tolist(), pred_all[np.asarray(test_idx, dtype=int)].tolist(), strict=False)
    ]
    return ClassificationRunResult(val_accuracy=val_acc, test_predictions=test_predictions)


def _symmetrize(adjacency: csr_matrix) -> csr_matrix:
    """将邻接矩阵转为无向对称图。"""
    return adjacency.maximum(adjacency.transpose()).tocsr()
