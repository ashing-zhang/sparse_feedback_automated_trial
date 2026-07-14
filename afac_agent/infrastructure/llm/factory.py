"""
运行指南：
- 本模块根据配置创建 LLMClient，不直接运行。
"""

from __future__ import annotations

import logging
import os

from afac_agent.domain.ports.llm_client import LLMClient
from afac_agent.infrastructure.config.schema import LLMConfig
from afac_agent.infrastructure.llm.noop_client import NoopLLMClient
from afac_agent.infrastructure.llm.openai_compatible_client import OpenAICompatibleLLMClient

logger = logging.getLogger(__name__)


def build_llm_client(cfg: LLMConfig) -> LLMClient:
    """根据配置创建 LLMClient。"""
    if not cfg.enabled:
        logger.info("LLM disabled by config")
        return NoopLLMClient()
    if not cfg.model.strip():
        logger.info("LLM enabled but llm.model is empty, fallback to NoopLLMClient")
        return NoopLLMClient()
    if not os.environ.get(cfg.api_key_env, "").strip():
        logger.info("LLM enabled but missing api key env=%s, fallback to NoopLLMClient", cfg.api_key_env)
        return NoopLLMClient()

    provider = cfg.provider.strip().lower()
    if provider == "openai_compatible":
        return OpenAICompatibleLLMClient(
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_seconds=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
        )

    raise ValueError(f"unknown llm provider: {cfg.provider!r}")

