"""
运行指南：
- 本模块实现 JSONL 形式的实验记录存储，不直接运行。
- 由 Experiment Agent 负责写入结构化实验反馈到 outputs/experiments.jsonl。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging

from afac_agent.domain.ports.experiment_store import ExperimentStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JsonlExperimentStore(ExperimentStore):
    """JSONL 实验记录存储。"""

    path: Path

    def append(self, record: dict[str, Any]) -> None:
        """追加一条实验记录。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info("Experiment record appended: %s", self.path)

