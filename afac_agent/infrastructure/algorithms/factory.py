"""
运行指南：
- 本模块负责根据配置字典实例化算法对象，不直接运行。
- 新增算法时：在对应目录实现类，并在此处注册 kind -> 工厂逻辑即可。
"""

from __future__ import annotations

from typing import Any

from afac_agent.domain.ports.algorithms import ClassificationAlgorithm, RecommendationAlgorithm
from afac_agent.infrastructure.algorithms.classification.gcn import GCNAlgorithm
from afac_agent.infrastructure.algorithms.classification.graph_transformer import GraphTransformerAlgorithm
from afac_agent.infrastructure.algorithms.classification.graphsage import GraphSAGEAlgorithm
from afac_agent.infrastructure.algorithms.classification.label_propagation import LabelPropagationAlgorithm
from afac_agent.infrastructure.algorithms.classification.logistic_regression import LogisticRegressionAlgorithm
from afac_agent.infrastructure.algorithms.recommendation.bert4rec import BERT4RecAlgorithm
from afac_agent.infrastructure.algorithms.recommendation.graph_transformer import GraphTransformerAlgorithm
from afac_agent.infrastructure.algorithms.recommendation.gru4rec import GRU4RecAlgorithm
from afac_agent.infrastructure.algorithms.recommendation.lightgcn import LightGCNAlgorithm



def build_classification_algorithm(config: dict[str, Any]) -> ClassificationAlgorithm:
    """根据配置创建分类任务算法实现。"""
    kind = str(config.get("kind", "")).strip()
    if kind == "gcn":
        return GCNAlgorithm(
            hidden_dim=int(config.get("hidden_dim", 64)),
            dropout=float(config.get("dropout", 0.5)),
            learning_rate=float(config.get("learning_rate", 0.01)),
            weight_decay=float(config.get("weight_decay", 5.0e-4)),
            epochs=int(config.get("epochs", 200)),
            seed=int(config.get("seed", 42)),
            patience=int(config.get("patience", 5)),
        )
    if kind == "graphsage":
        return GraphSAGEAlgorithm(
            hidden_dim=int(config.get("hidden_dim", 64)),
            dropout=float(config.get("dropout", 0.5)),
            learning_rate=float(config.get("learning_rate", 0.01)),
            weight_decay=float(config.get("weight_decay", 5.0e-4)),
            epochs=int(config.get("epochs", 200)),
            seed=int(config.get("seed", 42)),
            patience=int(config.get("patience", 5)),
        )
    if kind == "label_propagation":
        return LabelPropagationAlgorithm(
            alpha=float(config.get("alpha", 0.9)),
            max_iter=int(config.get("max_iter", 100)),
        )
    if kind == "logistic_regression":
        return LogisticRegressionAlgorithm(
            c=float(config.get("c", 1.0)),
            max_iter=int(config.get("max_iter", 200)),
        )
    if kind == "graph_transformer":
        return GraphTransformerAlgorithm(
            hidden_dim=int(config.get("hidden_dim", 64)),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=int(config.get("num_heads", 4)),
            dropout=float(config.get("dropout", 0.5)),
            learning_rate=float(config.get("learning_rate", 0.01)),
            weight_decay=float(config.get("weight_decay", 5.0e-4)),
            epochs=int(config.get("epochs", 200)),
            seed=int(config.get("seed", 42)),
            patience=int(config.get("patience", 5)),
        )
    raise ValueError(f"unknown classification algorithm kind: {kind!r}")


def build_recommendation_algorithm(config: dict[str, Any]) -> RecommendationAlgorithm:
    """根据配置创建推荐任务算法实现。"""
    kind = str(config.get("kind", "")).strip()
    if kind == "gru4rec":
        return GRU4RecAlgorithm(
            hidden_dim=int(config.get("hidden_dim", 64)),
            dropout=float(config.get("dropout", 0.5)),
            learning_rate=float(config.get("learning_rate", 0.01)),
            weight_decay=float(config.get("weight_decay", 5.0e-4)),
            epochs=int(config.get("epochs", 200)),
            seed=int(config.get("seed", 42)),
            top_k=10,
            patience=int(config.get("patience", 5)),
        )
    if kind == "lightgcn":
        return LightGCNAlgorithm(
            hidden_dim=int(config.get("hidden_dim", 64)),
            num_layers=int(config.get("num_layers", 2)),
            learning_rate=float(config.get("learning_rate", 0.01)),
            weight_decay=float(config.get("weight_decay", 5.0e-4)),
            epochs=int(config.get("epochs", 200)),
            seed=int(config.get("seed", 42)),
            top_k=10,
            patience=int(config.get("patience", 5)),
        )
    if kind == "graph_transformer":
        return GraphTransformerAlgorithm(
            hidden_dim=int(config.get("hidden_dim", 64)),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=int(config.get("num_heads", 4)),
            dropout=float(config.get("dropout", 0.5)),
            learning_rate=float(config.get("learning_rate", 0.01)),
            weight_decay=float(config.get("weight_decay", 5.0e-4)),
            epochs=int(config.get("epochs", 200)),
            seed=int(config.get("seed", 42)),
            top_k=10,
            patience=int(config.get("patience", 5)),
        )
    if kind == "bert4rec":
        return BERT4RecAlgorithm(
            hidden_dim=int(config.get("hidden_dim", 64)),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=int(config.get("num_heads", 4)),
            dropout=float(config.get("dropout", 0.5)),
            learning_rate=float(config.get("learning_rate", 0.01)),
            weight_decay=float(config.get("weight_decay", 5.0e-4)),
            epochs=int(config.get("epochs", 200)),
            seed=int(config.get("seed", 42)),
            top_k=10,
            patience=int(config.get("patience", 5)),
        )
    raise ValueError(f"unknown recommendation algorithm kind: {kind!r}")
