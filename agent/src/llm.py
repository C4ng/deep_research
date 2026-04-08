import logging
import os
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import openai
from openai import AsyncOpenAI, OpenAI
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Accumulated token usage across LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, prompt: int, completion: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += prompt + completion
            self.call_count += 1

    def to_dict(self) -> dict[str, int]:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "llm_calls": self.call_count,
            }

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.call_count = 0


def _log_retry(retry_state: RetryCallState) -> None:
    """Log a debug line whenever tenacity is about to retry."""
    fn_name = getattr(retry_state.fn, "__name__", "unknown")
    attempt = retry_state.attempt_number
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    sleep = retry_state.next_action.sleep if retry_state.next_action else None
    if sleep is not None:
        logger.warning("LLM:%s retry #%d after error: %s (sleep %.1fs)", fn_name, attempt, exc, sleep)
    else:
        logger.warning("LLM:%s retry #%d after error: %s", fn_name, attempt, exc)


class LLM:
    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        **kwargs,
    ):
        self.model_id = model_id or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")

        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = top_p or float(os.getenv("TOP_P", "1.0"))
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "2048"))
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))

        self._request_times: deque[float] = deque(maxlen=60)
        self._min_interval = 0.1
        self.usage = UsageStats()

        self.kwargs = kwargs

        if not self.api_key or not self.base_url:
            raise ValueError("API key and Base URL are required.")

        self.client = self._create_client()

    def _create_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def _create_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    _rate_limit_retry = retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
            )
        ),
        before_sleep=_log_retry,
        reraise=True,
    )

    _standard_retry = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (
                openai.APITimeoutError,
                openai.InternalServerError,
            )
        ),
        before_sleep=_log_retry,
        reraise=True,
    )

    def _handle_llm_exception(self, phase: str, exc: Exception) -> None:
        """Centralized exception diagnostics for both generate() and stream()."""
        error_type = type(exc).__name__
        error_msg = str(exc)

        is_rate_limit = (
            isinstance(exc, openai.RateLimitError)
            or "rate limit" in error_msg.lower()
            or "429" in error_msg
            or "quota" in error_msg.lower()
        )

        is_connection_error = (
            isinstance(exc, openai.APIConnectionError)
            or "connection" in error_type.lower()
            or "connection" in error_msg.lower()
            or "nodename nor servname" in error_msg.lower()
        )

        if is_rate_limit:
            logger.warning("LLM:%s rate limit error: %s", phase, error_msg)
        elif is_connection_error:
            logger.warning("LLM:%s connection error: %s", phase, error_msg)
        else:
            logger.error("LLM:%s error %s: %s", phase, error_type, error_msg)

        if is_connection_error or is_rate_limit:
            logger.debug(
                "LLM:%s config — base_url=%s model=%s api_key=%s timeout=%ds",
                phase,
                self.base_url,
                self.model_id,
                "SET" if self.api_key else "NOT SET",
                self.timeout,
            )

        logger.debug("LLM:%s full exception", phase, exc_info=exc)
        raise

    @_rate_limit_retry
    @_standard_retry
    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate a non-streaming response from the LLM."""
        self._throttle_requests()

        params = self._get_params(kwargs)
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                **params,
            )
            if response.usage:
                self.usage.record(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                )
            if not response.choices:
                return ""
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                logger.warning("LLM:generate response truncated due to max_tokens")
            return response.choices[0].message.content or ""
        except Exception as e:
            self._handle_llm_exception("generate", e)
            return ""  # unreachable; _handle_llm_exception always raises

    @_rate_limit_retry
    @_standard_retry
    def stream(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """Stream response fragments from the LLM."""
        self._throttle_requests()

        params = self._get_params({**kwargs, "stream": True})
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                **params,
            )

            for chunk in response:  # type: ignore[union-attr]
                if hasattr(chunk, "usage") and chunk.usage:  # type: ignore[attr-defined]
                    self.usage.record(
                        getattr(chunk.usage, "prompt_tokens", 0) or 0,  # type: ignore[attr-defined]
                        getattr(chunk.usage, "completion_tokens", 0) or 0,  # type: ignore[attr-defined]
                    )

                if not chunk.choices:  # type: ignore[attr-defined]
                    continue

                delta = chunk.choices[0].delta  # type: ignore[attr-defined]

                if hasattr(delta, "content") and delta.content:
                    yield delta.content

                if chunk.choices[0].finish_reason == "content_filter":  # type: ignore[attr-defined]
                    logger.warning("LLM:stream content omitted due to safety filters")
                    break
                elif chunk.choices[0].finish_reason == "length":  # type: ignore[attr-defined]
                    logger.warning("LLM:stream response truncated due to max_tokens")
                    break

        except Exception as e:
            self._handle_llm_exception("stream", e)

    def _get_params(self, overrides: dict[str, Any]) -> dict[str, Any]:
        base = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        merged = {**base, **self.kwargs, **overrides}
        return {k: v for k, v in merged.items() if v is not None}

    def _throttle_requests(self) -> None:
        """Throttle requests to avoid triggering rate limits."""
        now = time.time()

        if self._request_times:
            last_request_time = self._request_times[-1]
            elapsed = now - last_request_time

            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                time.sleep(wait_time)

        self._request_times.append(time.time())
