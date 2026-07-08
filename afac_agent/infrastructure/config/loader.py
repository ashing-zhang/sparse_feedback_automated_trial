"""
运行指南：
- 本文件提供配置加载能力，不直接运行。
- 入口模块会读取 AFAC_CONFIG_PATH（默认 configs/default.yaml）并使用本模块加载配置。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from afac_agent.infrastructure.config.schema import (
    AppConfig,
    ClassificationDataConfig,
    DataConfig,
    ExperimentConfig,
    RecommendationDataConfig,
    RunConfig,
    SearchSpaceConfig,
    SubmissionConfig,
)


def load_yaml(path: Path) -> dict[str, Any]:
    """从 YAML 文件读取配置字典。"""
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config yaml must be a mapping, got: {type(data)}")
    return data


def _require(mapping: dict[str, Any], key: str) -> Any:
    """从字典中读取必填字段。"""
    if key not in mapping:
        raise ValueError(f"missing required config key: {key}")
    return mapping[key]


def _as_path(value: Any) -> Path:
    """将值解析为 Path。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"path must be a non-empty string, got: {value!r}")
    return Path(value)


def parse_config(data: dict[str, Any]) -> AppConfig:
    """解析配置字典为类型化 AppConfig。"""
    run = _require(data, "run")
    if not isinstance(run, dict):
        raise ValueError("run must be a mapping")

    exp = _require(run, "experiment")
    if not isinstance(exp, dict):
        raise ValueError("run.experiment must be a mapping")

    run_cfg = RunConfig(
        seed=int(_require(run, "seed")),
        output_dir=_as_path(_require(run, "output_dir")),
        experiment=ExperimentConfig(
            max_trials=int(_require(exp, "max_trials")),
            patience=int(_require(exp, "patience")),
        ),
    )

    data_block = _require(data, "data")
    if not isinstance(data_block, dict):
        raise ValueError("data must be a mapping")

    cls = _require(data_block, "classification")
    rec = _require(data_block, "recommendation")
    if not isinstance(cls, dict) or not isinstance(rec, dict):
        raise ValueError("data.classification and data.recommendation must be mappings")

    data_cfg = DataConfig(
        classification=ClassificationDataConfig(
            npz_path=_as_path(_require(cls, "npz_path")),
            sample_submission_path=_as_path(_require(cls, "sample_submission_path")),
            val_ratio=float(_require(cls, "val_ratio")),
        ),
        recommendation=RecommendationDataConfig(
            train_csv_path=_as_path(_require(rec, "train_csv_path")),
            test_csv_path=_as_path(_require(rec, "test_csv_path")),
            user_csv_path=_as_path(_require(rec, "user_csv_path")),
            item_csv_path=_as_path(_require(rec, "item_csv_path")),
            sample_submission_path=_as_path(_require(rec, "sample_submission_path")),
            val_ratio=float(_require(rec, "val_ratio")),
        ),
    )

    search = _require(data, "search_space")
    if not isinstance(search, dict):
        raise ValueError("search_space must be a mapping")
    search_cls = _require(search, "classification")
    search_rec = _require(search, "recommendation")
    if not isinstance(search_cls, dict) or not isinstance(search_rec, dict):
        raise ValueError("search_space.classification and search_space.recommendation must be mappings")
    cls_candidates = _require(search_cls, "candidates")
    rec_candidates = _require(search_rec, "candidates")
    if not isinstance(cls_candidates, list) or not isinstance(rec_candidates, list):
        raise ValueError("search_space.*.candidates must be lists")
    search_cfg = SearchSpaceConfig(
        classification_candidates=[dict(x) for x in cls_candidates],
        recommendation_candidates=[dict(x) for x in rec_candidates],
    )

    sub = _require(data, "submission")
    if not isinstance(sub, dict):
        raise ValueError("submission must be a mapping")
    sub_cfg = SubmissionConfig(
        a1_filename=str(_require(sub, "a1_filename")),
        a2_filename=str(_require(sub, "a2_filename")),
        zip_filename=str(_require(sub, "zip_filename")),
    )

    return AppConfig(run=run_cfg, data=data_cfg, search_space=search_cfg, submission=sub_cfg)


def load_config(path: Path) -> AppConfig:
    """从 YAML 文件加载并解析配置。"""
    return parse_config(load_yaml(path))


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """将配置转换为可序列化字典（便于记录实验）。"""
    return asdict(config)
