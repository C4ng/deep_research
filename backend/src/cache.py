from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Tuple
import hashlib
import json
import os
from pathlib import Path


class LLMCache:
    """Thread-safe in-memory LRU cache with optional TTL for LLM calls."""

    def __init__(self, max_size: int = 256, ttl_seconds: float = 3600) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[Tuple[str, str], Tuple[float, Any]] = OrderedDict()
        self._lock = Lock()

    def _make_key(self, namespace: str, prompt: str) -> Tuple[str, str]:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return namespace, digest

    def get(self, namespace: str, prompt: str) -> Any | None:
        key = self._make_key(namespace, prompt)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if self._ttl > 0 and (time.monotonic() - ts) > self._ttl:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, namespace: str, prompt: str, value: Any) -> None:
        key = self._make_key(namespace, prompt)
        with self._lock:
            self._store[key] = (time.monotonic(), value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)


# Global cache instance used across planner / reviewer / summarizer / reporter.
llm_cache = LLMCache()


class StageFileCache:
    """Filesystem cache for stage outputs keyed by research topic."""

    def __init__(self, base_dir: str = ".cache/deep_research") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _enabled(self) -> bool:
        # Can be disabled for tests or debugging.
        return os.getenv("DEEP_RESEARCH_FILE_CACHE", "1") == "1"

    def _topic_dir(self, topic: str) -> Path:
        digest = hashlib.sha256((topic or "").encode("utf-8")).hexdigest()[:16]
        return self._base_dir / digest

    def _stage_file(self, topic: str, stage: str, task_id: int | None = None) -> Path:
        topic_dir = self._topic_dir(topic)
        topic_dir.mkdir(parents=True, exist_ok=True)
        if task_id is None:
            return topic_dir / f"{stage}.json"
        return topic_dir / f"{stage}_task_{task_id}.json"

    def load(self, topic: str, stage: str, task_id: int | None = None) -> dict[str, Any] | None:
        if not self._enabled():
            return None
        path = self._stage_file(topic, stage, task_id)
        if not path.exists():
            return None
        with self._lock:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None

    def save(self, topic: str, stage: str, payload: dict[str, Any], task_id: int | None = None) -> None:
        if not self._enabled():
            return
        path = self._stage_file(topic, stage, task_id)
        with self._lock:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# Global file cache for planner/search/reviewer/summary/report stages.
stage_file_cache = StageFileCache()

