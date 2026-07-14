"""
运行指南：
- 本模块提供 Prompt 文件加载能力，不直接运行。
- Prompt 内容建议以纯文本/Markdown 存放于 configs/prompts/ 并通过配置引用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PromptLoader:
    """Prompt 加载器。"""

    def load(self, path: Path) -> str:
        """加载 prompt 文本。"""
        return path.read_text(encoding="utf-8").strip()

