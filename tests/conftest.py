"""Shared fixtures and test configuration for the deep_research test suite."""

from __future__ import annotations

import os

import pytest

from backend.src.config import Configuration
from backend.src.models import SummaryState, TodoItem, TaskReview


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: tests that hit real LLM / network services")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set test-mode env vars: disable filesystem cache and debug log file."""
    monkeypatch.setenv("DEEP_RESEARCH_FILE_CACHE", "0")
    monkeypatch.setenv("TESTING", "1")


@pytest.fixture(autouse=True)
def _clear_llm_cache() -> None:
    """Reset the global in-memory LLM cache between tests to avoid cross-contamination."""
    from backend.src.cache import llm_cache
    with llm_cache._lock:
        llm_cache._store.clear()


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_state() -> SummaryState:
    return SummaryState(research_topic="Test topic")


@pytest.fixture()
def sample_task() -> TodoItem:
    return TodoItem(id=1, title="T1", intent="I1", query="Q1")


@pytest.fixture()
def passing_review() -> TaskReview:
    return TaskReview(
        coverage_score=0.9,
        reliability_score=0.9,
        clarity_score=0.9,
        overall_score=0.9,
        should_reresearch=False,
        recommendations=[],
        notes="Looks good",
    )


@pytest.fixture()
def failing_review() -> TaskReview:
    return TaskReview(
        coverage_score=0.4,
        reliability_score=0.4,
        clarity_score=0.4,
        overall_score=0.4,
        should_reresearch=True,
        recommendations=["Need more data"],
        notes="Weak coverage",
    )


# ---------------------------------------------------------------------------
# Fake agent helpers
# ---------------------------------------------------------------------------

class FakeAgent:
    """Minimal stand-in for SimpleAgent. Supports single or sequential responses."""

    def __init__(self, response: str | list[str] = ""):
        self._responses = response if isinstance(response, list) else [response]
        self._call_idx = 0
        self.last_prompt: str | None = None

    def run(self, prompt: str) -> str:
        self.last_prompt = prompt
        if self._call_idx < len(self._responses):
            result = self._responses[self._call_idx]
            self._call_idx += 1
            return result
        return self._responses[-1] if self._responses else ""

    def clear_history(self) -> None:
        pass


class FakeSearchTool:
    """Minimal stand-in for SearchTool."""

    def __init__(self, response: dict | str):
        self._response = response
        self.last_payload: dict | None = None

    def run(self, payload: dict) -> dict | str:
        self.last_payload = payload
        return self._response


@pytest.fixture()
def fake_agent():
    return FakeAgent


@pytest.fixture()
def fake_search_tool():
    return FakeSearchTool
