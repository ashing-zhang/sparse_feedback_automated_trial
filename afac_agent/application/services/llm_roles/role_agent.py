"""
运行指南：
- 本模块定义多角色 Agent 的基础委托封装，不直接运行。
- 各角色的 prompt 通过配置引用，便于在不改代码的情况下迭代提示词。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from afac_agent.domain.ports.llm_client import ChatMessage, LLMClient
from afac_agent.infrastructure.llm.prompt_loader import PromptLoader
from afac_agent.application.services.llm_utils.json_parsing import parse_json_object


@dataclass(frozen=True, slots=True)
class RoleAgent:
    """多角色 Agent 的通用委托。"""

    llm: LLMClient
    prompt_path: Path
    prompt_loader: PromptLoader

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行角色推理并返回 JSON object。"""
        system_prompt = self.prompt_loader.load(self.prompt_path)
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]
        text = self.llm.chat(messages)
        return parse_json_object(text)

