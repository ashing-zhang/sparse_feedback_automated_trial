"""
运行指南：
- 本文件提供一个轻量 .env 加载器，不依赖 python-dotenv。
- 入口模块会在读取环境变量前调用 load_dotenv_if_present()。
"""

from __future__ import annotations

from pathlib import Path
import os


def load_dotenv_if_present(path: Path) -> None:
    """如存在 .env 文件，则将其中 KEY=VALUE 写入环境变量（已存在则不覆盖）。"""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ.setdefault(key, value)

