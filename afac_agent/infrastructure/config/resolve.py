"""
运行指南：
- 本模块负责将配置中的相对路径解析为绝对路径，不直接运行。
- 入口模块会在加载 YAML 配置后调用 resolve_paths()。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from afac_agent.infrastructure.config.schema import (
    AgentPromptsConfig,
    AppConfig,
    ClassificationDataConfig,
    DataConfig,
    RecommendationDataConfig,
    RunConfig,
)


def resolve_paths(config: AppConfig, base_dir: Path) -> AppConfig:
    """将配置中的 Path 字段按 base_dir 解析为绝对路径。"""
    run = config.run
    data = config.data

    resolved_run = RunConfig(
        seed=run.seed,
        output_dir=_resolve_path(run.output_dir, base_dir),
        experiment=run.experiment,
    )

    resolved_cls = ClassificationDataConfig(
        npz_path=_resolve_path(data.classification.npz_path, base_dir),
        sample_submission_path=_resolve_path(data.classification.sample_submission_path, base_dir),
        val_ratio=data.classification.val_ratio,
    )
    resolved_rec = RecommendationDataConfig(
        train_csv_path=_resolve_path(data.recommendation.train_csv_path, base_dir),
        test_csv_path=_resolve_path(data.recommendation.test_csv_path, base_dir),
        user_csv_path=_resolve_path(data.recommendation.user_csv_path, base_dir),
        item_csv_path=_resolve_path(data.recommendation.item_csv_path, base_dir),
        sample_submission_path=_resolve_path(data.recommendation.sample_submission_path, base_dir),
        val_ratio=data.recommendation.val_ratio,
    )

    resolved_data = DataConfig(classification=resolved_cls, recommendation=resolved_rec)
    prompts = config.agent_prompts
    resolved_prompts = AgentPromptsConfig(
        planner_path=_resolve_path(prompts.planner_path, base_dir),
        scientist_path=_resolve_path(prompts.scientist_path, base_dir),
        engineer_path=_resolve_path(prompts.engineer_path, base_dir),
        reviewer_path=_resolve_path(prompts.reviewer_path, base_dir),
    )
    return replace(config, run=resolved_run, data=resolved_data, agent_prompts=resolved_prompts)


def _resolve_path(path: Path, base_dir: Path) -> Path:
    """解析路径：绝对路径直接返回；相对路径按 base_dir 拼接。"""
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
