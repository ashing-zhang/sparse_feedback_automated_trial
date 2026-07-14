"""
运行指南：
- 本文件为领域层端口（Protocol）定义，不直接运行。
- 基础设施层提供具体的 LLM 客户端实现（如 OpenAI Compatible HTTP API）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Literal

ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """LLM 对话消息。"""

    role: ChatRole
    content: str


class LLMClient(Protocol):
    """LLM 客户端端口。"""

    def chat(self, messages: list[ChatMessage]) -> str:
        """执行一次对话补全并返回 assistant 文本。"""

