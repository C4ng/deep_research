"""Unit tests for the caching layer (LLMCache + StageFileCache)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.src.cache import LLMCache, StageFileCache


# ---------------------------------------------------------------------------
# LLMCache
# ---------------------------------------------------------------------------

class TestLLMCache:

    def test_get_returns_none_on_miss(self):
        cache = LLMCache()
        assert cache.get("ns", "prompt") is None

    def test_set_then_get_returns_value(self):
        cache = LLMCache()
        cache.set("ns", "prompt", "value")
        assert cache.get("ns", "prompt") == "value"

    def test_namespace_isolation(self):
        cache = LLMCache()
        cache.set("reviewer", "p1", "r1")
        cache.set("summarizer", "p1", "s1")
        assert cache.get("reviewer", "p1") == "r1"
        assert cache.get("summarizer", "p1") == "s1"

    def test_overwrite_existing_key(self):
        cache = LLMCache()
        cache.set("ns", "p", "old")
        cache.set("ns", "p", "new")
        assert cache.get("ns", "p") == "new"

    def test_different_prompts_different_keys(self):
        cache = LLMCache()
        cache.set("ns", "prompt_a", "val_a")
        cache.set("ns", "prompt_b", "val_b")
        assert cache.get("ns", "prompt_a") == "val_a"
        assert cache.get("ns", "prompt_b") == "val_b"


# ---------------------------------------------------------------------------
# StageFileCache
# ---------------------------------------------------------------------------

class TestStageFileCache:

    @pytest.fixture(autouse=True)
    def _enable_file_cache(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DEEP_RESEARCH_FILE_CACHE", "1")

    def test_save_and_load(self, tmp_path: Path):
        cache = StageFileCache(base_dir=str(tmp_path / "cache"))
        payload = {"key": "value", "num": 42}
        cache.save("topic1", "planner", payload)
        loaded = cache.load("topic1", "planner")
        assert loaded == payload

    def test_save_and_load_with_task_id(self, tmp_path: Path):
        cache = StageFileCache(base_dir=str(tmp_path / "cache"))
        cache.save("topic1", "review", {"score": 0.8}, task_id=1)
        cache.save("topic1", "review", {"score": 0.6}, task_id=2)
        assert cache.load("topic1", "review", task_id=1)["score"] == 0.8
        assert cache.load("topic1", "review", task_id=2)["score"] == 0.6

    def test_load_returns_none_on_miss(self, tmp_path: Path):
        cache = StageFileCache(base_dir=str(tmp_path / "cache"))
        assert cache.load("nonexistent", "planner") is None

    def test_load_returns_none_for_corrupt_json(self, tmp_path: Path):
        cache = StageFileCache(base_dir=str(tmp_path / "cache"))
        cache.save("topic", "stage", {"ok": True})
        path = cache._stage_file("topic", "stage")
        path.write_text("NOT VALID JSON", encoding="utf-8")
        assert cache.load("topic", "stage") is None

    def test_disabled_via_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DEEP_RESEARCH_FILE_CACHE", "0")
        cache = StageFileCache(base_dir=str(tmp_path / "cache"))
        cache.save("topic", "planner", {"data": 1})
        assert cache.load("topic", "planner") is None

    def test_different_topics_isolated(self, tmp_path: Path):
        cache = StageFileCache(base_dir=str(tmp_path / "cache"))
        cache.save("topic_a", "planner", {"a": 1})
        cache.save("topic_b", "planner", {"b": 2})
        assert cache.load("topic_a", "planner") == {"a": 1}
        assert cache.load("topic_b", "planner") == {"b": 2}
