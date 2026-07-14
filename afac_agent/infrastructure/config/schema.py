"""
运行指南：
- 本文件为配置数据模型定义，不直接运行。
- 入口模块会读取 YAML 配置并解析为这些 dataclass，再用于组装系统。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """实验控制配置。"""

    max_trials: int
    patience: int


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """LLM 配置（OpenAI Compatible）。"""

    enabled: bool
    provider: str
    base_url: str
    api_key_env: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class AgentPromptsConfig:
    """多角色 Agent Prompt 配置。"""

    planner_path: Path
    scientist_path: Path
    engineer_path: Path
    reviewer_path: Path


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """实验记忆系统配置。"""

    mode: str
    recent_k: int
    long_term_k: int


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """串行优化控制器配置。"""

    explore_ratio: float
    optimize_ratio: float
    stabilize_ratio: float
    improvement_threshold: float
    max_rounds: int | None


@dataclass(frozen=True, slots=True)
class RunConfig:
    """运行配置。"""

    seed: int
    output_dir: Path
    experiment: ExperimentConfig


@dataclass(frozen=True, slots=True)
class ClassificationDataConfig:
    """分类任务数据配置。"""

    npz_path: Path
    sample_submission_path: Path
    val_ratio: float


@dataclass(frozen=True, slots=True)
class RecommendationDataConfig:
    """推荐任务数据配置。"""

    train_csv_path: Path
    test_csv_path: Path
    user_csv_path: Path
    item_csv_path: Path
    sample_submission_path: Path
    val_ratio: float


@dataclass(frozen=True, slots=True)
class DataConfig:
    """数据配置。"""

    classification: ClassificationDataConfig
    recommendation: RecommendationDataConfig


@dataclass(frozen=True, slots=True)
class SearchSpaceConfig:
    """搜索空间配置（候选实验列表）。"""

    classification_candidates: list[dict[str, Any]]
    recommendation_candidates: list[dict[str, Any]]
    classification_mutation_rules: dict[str, Any]
    recommendation_mutation_rules: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SubmissionConfig:
    """提交文件配置。"""

    a1_filename: str
    a2_filename: str
    zip_filename: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    """应用总配置。"""

    run: RunConfig
    data: DataConfig
    search_space: SearchSpaceConfig
    submission: SubmissionConfig
    llm: LLMConfig
    agent_prompts: AgentPromptsConfig
    memory: MemoryConfig
    controller: ControllerConfig
