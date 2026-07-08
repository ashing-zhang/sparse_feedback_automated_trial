"""
运行指南：
- 本模块负责写出 A1.csv / A2.csv 并打包为 prediction.zip，不直接运行。
- 由入口模块组装为 SubmissionWriter 端口实现供应用层调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import zipfile

import pandas as pd

from afac_agent.domain.models.predictions import A1PredictionRow, A2PredictionRow
from afac_agent.infrastructure.config.schema import DataConfig, SubmissionConfig
from afac_agent.domain.ports.submission_writer import SubmissionWriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FileSubmissionWriter(SubmissionWriter):
    """基于文件系统的提交文件写出实现。"""

    output_dir: Path
    data_config: DataConfig
    submission_config: SubmissionConfig

    def write_a1(self, rows: list[A1PredictionRow]) -> str:
        """写出 A1.csv（严格按 sample_submission 的 test_idx 顺序）。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / self.submission_config.a1_filename

        sample = pd.read_csv(self.data_config.classification.sample_submission_path)
        if "test_idx" not in sample.columns:
            raise ValueError("A1 sample_submission.csv missing column: test_idx")

        pred_map = {int(r.test_idx): int(r.label) for r in rows}
        ordered_idx = sample["test_idx"].astype(int).tolist()

        missing = [i for i in ordered_idx if i not in pred_map]
        if missing:
            raise ValueError(f"A1 missing predictions for test_idx: {missing[:10]} (total={len(missing)})")

        out = pd.DataFrame({"test_idx": ordered_idx, "label": [pred_map[i] for i in ordered_idx]})
        out.to_csv(path, index=False, encoding="utf-8")
        logger.info("Wrote A1 submission: %s", path)
        return str(path)

    def write_a2(self, rows: list[A2PredictionRow]) -> str:
        """写出 A2.csv（严格按 sample_submission 的 uid 顺序，且每行 10 个 iid）。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / self.submission_config.a2_filename

        sample = pd.read_csv(self.data_config.recommendation.sample_submission_path)
        if "uid" not in sample.columns:
            raise ValueError("A2 sample_submission.csv missing column: uid")

        items_df = pd.read_csv(self.data_config.recommendation.item_csv_path, usecols=["iid"])
        item_universe = set(items_df["iid"].astype(str).tolist())
        pred_map = {str(r.uid): list(r.prediction) for r in rows}
        ordered_uid = sample["uid"].astype(str).tolist()

        missing = [u for u in ordered_uid if u not in pred_map]
        if missing:
            raise ValueError(f"A2 missing predictions for uid: {missing[:10]} (total={len(missing)})")

        predictions: list[str] = []
        for uid in ordered_uid:
            items = [str(x) for x in pred_map[uid]]
            items = _sanitize_prediction(items, item_universe)
            if len(items) != 10:
                raise ValueError(f"A2 prediction length must be 10, uid={uid}, got={len(items)}")
            predictions.append(",".join(items))

        out = pd.DataFrame({"uid": ordered_uid, "prediction": predictions})
        out.to_csv(path, index=False, encoding="utf-8")
        logger.info("Wrote A2 submission: %s", path)
        return str(path)

    def write_zip(self, a1_path: str, a2_path: str) -> str:
        """将 A1/A2 打包为 prediction.zip。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.output_dir / self.submission_config.zip_filename

        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(a1_path, arcname=Path(a1_path).name)
            zf.write(a2_path, arcname=Path(a2_path).name)

        logger.info("Wrote prediction zip: %s", zip_path)
        return str(zip_path)


def _sanitize_prediction(items: list[str], item_universe: set[str]) -> list[str]:
    """去重、过滤非法 iid，并保持顺序。"""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        if x not in item_universe:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= 10:
            break
    if len(out) >= 10:
        return out
    for x in sorted(item_universe):
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= 10:
            break
    return out
