"""
运行指南：
1) 复制 .env.example 为 .env，按需修改 AFAC_CONFIG_PATH / AFAC_LOGGING_PATH
2) 运行：
   python -m afac_agent.run

说明：
- 不使用 argparse；配置文件与日志配置文件路径从环境变量读取
- 默认配置会读取 data/ 目录的 A 榜数据，并输出到 outputs/
"""

from __future__ import annotations

from pathlib import Path
import os
import logging

from afac_agent.application.services.experiment_controller import ExperimentController
from afac_agent.application.services.experiment_agent import ExperimentAgent
from afac_agent.application.services.llm_experiment_policy import LLMExperimentPolicy
from afac_agent.application.services.llm_roles.role_agent import RoleAgent
from afac_agent.application.services.memory_system import MemorySystem
from afac_agent.application.services.task_analyzer import TaskAnalyzer
from afac_agent.infrastructure.config.dotenv import load_dotenv_if_present
from afac_agent.infrastructure.config.loader import load_config
from afac_agent.infrastructure.config.resolve import resolve_paths
from afac_agent.infrastructure.datasets.file_dataset_loader import FileDatasetLoader
from afac_agent.infrastructure.experiments.jsonl_store import JsonlExperimentStore
from afac_agent.infrastructure.llm.factory import build_llm_client
from afac_agent.infrastructure.llm.prompt_loader import PromptLoader
from afac_agent.infrastructure.observability.logging_setup import setup_logging
from afac_agent.infrastructure.outputs.file_submission_writer import FileSubmissionWriter


def main() -> int:
    """程序入口：组装系统并生成 prediction.zip。"""
    base_dir = _repo_root()
    os.chdir(base_dir)
    load_dotenv_if_present(base_dir / ".env")

    config_path = _env_path("AFAC_CONFIG_PATH", base_dir / "configs" / "default.yaml")
    logging_path = _env_path("AFAC_LOGGING_PATH", base_dir / "configs" / "logging.yaml")

    (base_dir / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    setup_logging(logging_path)
    logger = logging.getLogger(__name__)

    cfg = resolve_paths(load_config(config_path), base_dir=base_dir)
    cfg.run.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_loader = FileDatasetLoader(data_config=cfg.data)
    submission_writer = FileSubmissionWriter(
        output_dir=cfg.run.output_dir,
        data_config=cfg.data,
        submission_config=cfg.submission,
    )
    experiment_store = JsonlExperimentStore(path=cfg.run.output_dir / "experiments.jsonl")

    llm = build_llm_client(cfg.llm)
    prompt_loader = PromptLoader()
    planner = RoleAgent(llm=llm, prompt_path=cfg.agent_prompts.planner_path, prompt_loader=prompt_loader)
    scientist = RoleAgent(llm=llm, prompt_path=cfg.agent_prompts.scientist_path, prompt_loader=prompt_loader)
    engineer = RoleAgent(llm=llm, prompt_path=cfg.agent_prompts.engineer_path, prompt_loader=prompt_loader)
    reviewer = RoleAgent(llm=llm, prompt_path=cfg.agent_prompts.reviewer_path, prompt_loader=prompt_loader)
    policy = LLMExperimentPolicy(
        planner=planner,
        scientist=scientist,
        engineer=engineer,
        reviewer=reviewer,
        improvement_threshold=cfg.controller.improvement_threshold,
        patience=cfg.run.experiment.patience,
    )
    analyzer = TaskAnalyzer()
    memory = MemorySystem(store=experiment_store, config=cfg.memory)
    controller = ExperimentController(
        config=cfg,
        policy=policy,
        store=experiment_store,
        analyzer=analyzer,
        memory=memory,
    )

    agent = ExperimentAgent(
        config=cfg,
        dataset_loader=dataset_loader,
        submission_writer=submission_writer,
        experiment_store=experiment_store,
        controller=controller,
    )
    zip_path = agent.run_all()
    logger.info("Done. prediction zip: %s", zip_path)
    return 0


def _repo_root() -> Path:
    """定位仓库根目录。"""
    return Path(__file__).resolve().parents[2]


def _env_path(key: str, default: Path) -> Path:
    """从环境变量读取路径，若无则返回默认值。"""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    return Path(raw)
