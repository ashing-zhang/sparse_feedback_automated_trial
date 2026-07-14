"""
运行指南：
- 本模块实现 OpenAI Compatible 的 Chat Completions HTTP 客户端，不直接运行。
- 通过配置 llm.base_url / llm.api_key_env / llm.model 启用；入口模块负责依赖注入。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from afac_agent.domain.ports.llm_client import ChatMessage, LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleLLMClient(LLMClient):
    """OpenAI Compatible LLMClient 实现（/chat/completions）。"""

    base_url: str
    api_key_env: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_retries: int

    def chat(self, messages: list[ChatMessage]) -> str:
        """执行一次对话补全并返回 assistant 文本。"""
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"missing api key in env: {self.api_key_env}")

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        url = _join_url(self.base_url, "/chat/completions")
        req = Request(
            url=url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        return _request_with_retry(
            req=req,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )


def _join_url(base_url: str, path: str) -> str:
    """拼接 base_url 与 path。"""
    b = base_url.rstrip("/")
    p = path.lstrip("/")
    return f"{b}/{p}"


def _request_with_retry(req: Request, timeout_seconds: float, max_retries: int) -> str:
    """带重试的 HTTP 请求并提取 assistant content。"""
    attempt = 0
    last_err: Exception | None = None
    while attempt <= max_retries:
        attempt += 1
        try:
            with urlopen(req, timeout=timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
            return _extract_content(body)
        except (HTTPError, URLError, TimeoutError, ValueError) as e:
            last_err = e
            logger.warning("LLM request failed (attempt=%d/%d): %s", attempt, max_retries + 1, e)
            if attempt > max_retries:
                break
            time.sleep(_backoff_seconds(attempt))
    raise RuntimeError(f"llm request failed after retries: {last_err}")


def _backoff_seconds(attempt: int) -> float:
    """指数退避秒数。"""
    return min(8.0, 0.5 * (2 ** max(0, attempt - 1)))


def _extract_content(raw_json: str) -> str:
    """从 OpenAI Compatible 响应中提取 choices[0].message.content。"""
    data: Any = json.loads(raw_json)
    if not isinstance(data, dict):
        raise ValueError("llm response must be a json object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("llm response missing choices")
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        raise ValueError("llm response missing message")
    content = msg.get("content")
    if not isinstance(content, str):
        raise ValueError("llm response missing content")
    return content

