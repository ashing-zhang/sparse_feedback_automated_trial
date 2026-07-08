"""
运行指南：
- 本文件提供 logging 初始化能力，不直接运行。
- 入口模块会读取 AFAC_LOGGING_PATH（默认 configs/logging.yaml）并调用 setup_logging()。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging.config

import yaml


def setup_logging(logging_config_path: Path) -> None:
    """根据 YAML 配置初始化 logging。"""
    raw = logging_config_path.read_text(encoding="utf-8")
    config: Any = yaml.safe_load(raw) or {}
    if not isinstance(config, dict):
        raise ValueError("logging config yaml must be a mapping")
    logging.config.dictConfig(config)

