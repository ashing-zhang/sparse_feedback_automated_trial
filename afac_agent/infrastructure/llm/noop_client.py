"""
运行指南：
- 本模块提供 Noop LLMClient，用于在未启用 LLM 或缺少密钥时保持系统可运行。
"""

from __future__ import annotations

from dataclasses import dataclass

from afac_agent.domain.ports.llm_client import ChatMessage, LLMClient


@dataclass(frozen=True, slots=True)
class NoopLLMClient(LLMClient):
    """不执行实际请求的 LLM 客户端。"""

    def chat(self, messages: list[ChatMessage]) -> str:
        """返回空字符串。"""
        return ""

