import pytest

import backend.src.agent as agent_module
from backend.src.agent import DeepResearchAgent
from backend.src.config import Configuration
from backend.src.models import SummaryState, TaskReview, TaskStatus, TodoItem


class _FakeReviewer:
    _fallback_marker = "[FALLBACK_REVIEW]"

    def __init__(self, reviews: list[TaskReview]):
        self._reviews = reviews
        self.calls = 0

    def review_task(self, state, task, context) -> TaskReview:
        review = self._reviews[min(self.calls, len(self._reviews) - 1)]
        self.calls += 1
        return review

    def is_fallback_review(self, review: TaskReview) -> bool:
        return isinstance(review.notes, str) and review.notes.startswith(self._fallback_marker)


class _FakeSummarizer:
    def __init__(self, summary: str = "SUMMARY", stream_chunks: list[str] | None = None):
        self.summary = summary
        self.stream_chunks = stream_chunks or ["chunk1", "chunk2"]

    def summarize_task(self, state, task, context) -> str:  # pragma: no cover - trivial
        return self.summary

    def stream_task_summary(self, state, task, context):
        def gen():
            yield from self.stream_chunks

        def getter():
            return "".join(self.stream_chunks)

        return gen(), getter


def _make_state_and_task() -> tuple[SummaryState, TodoItem]:
    state = SummaryState(research_topic="Test topic")
    task = TodoItem(
        id=1,
        title="T1",
        intent="I1",
        query="Q1",
    )
    return state, task


def _setup_agent(monkeypatch: pytest.MonkeyPatch, max_loops: int = 3) -> DeepResearchAgent:
    # Prevent LLM init from failing by providing dummy env vars.
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://test")
    # Disable persistent file cache so orchestration assertions remain deterministic.
    monkeypatch.setenv("DEEP_RESEARCH_FILE_CACHE", "0")

    config = Configuration(max_web_research_loops=max_loops)
    agent = DeepResearchAgent(config=config)
    return agent


def test_execute_task_single_loop_when_reviewer_satisfied(monkeypatch: pytest.MonkeyPatch):
    agent = _setup_agent(monkeypatch, max_loops=3)

    # Patch dispatch_search and prepare_research_context to be deterministic.
    calls: list[str] = []

    def fake_dispatch(query, search_backend, fetch_full_page, loop_count):
        calls.append(query)
        return (
            {"results": [{"title": "T", "url": "https://example.com", "content": "C"}]},
            [],
            None,
            "backend",
        )

    def fake_prepare(search_result, answer_text, fetch_full_page):
        return "SRC", "CTX"

    monkeypatch.setattr(agent_module, "dispatch_search", fake_dispatch)
    monkeypatch.setattr(agent_module, "prepare_research_context", fake_prepare)

    # Reviewer is happy on first loop.
    review = TaskReview(
        coverage_score=0.9,
        reliability_score=0.9,
        clarity_score=0.9,
        overall_score=0.9,
        should_reresearch=False,
        recommendations=[],
        notes="Looks good",
    )
    agent.reviewer = _FakeReviewer([review])
    agent.summarizer = _FakeSummarizer(summary="FINAL")

    state, task = _make_state_and_task()
    events = list(agent._execute_task(state, task, emit_stream=False, step=None))

    # No streaming events in non-streaming mode.
    assert events == []

    # Only one dispatch_search call.
    assert calls == ["Q1"]

    assert task.status == TaskStatus.COMPLETED
    assert task.summary == "FINAL"
    assert task.sources_summary == "SRC"


def test_execute_task_multi_loop_until_threshold(monkeypatch: pytest.MonkeyPatch):
    agent = _setup_agent(monkeypatch, max_loops=2)

    calls: list[str] = []
    loop_index = {"value": 0}

    def fake_dispatch(query, search_backend, fetch_full_page, loop_count):
        loop_index["value"] += 1
        calls.append(f"{query}-loop{loop_index['value']}")
        return (
            {"results": [{"title": f"T{loop_index['value']}", "url": "u", "content": "c"}]},
            [],
            None,
            "backend",
        )

    def fake_prepare(search_result, answer_text, fetch_full_page):
        # Use loop_index to generate distinct context/sources.
        idx = loop_index["value"]
        return f"SRC{idx}", f"CTX{idx}"

    monkeypatch.setattr(agent_module, "dispatch_search", fake_dispatch)
    monkeypatch.setattr(agent_module, "prepare_research_context", fake_prepare)

    # Reviewer will request a second loop, but the agent caps iterations to 2.
    reviews = [
        TaskReview(
            coverage_score=0.4,
            reliability_score=0.4,
            clarity_score=0.4,
            overall_score=0.4,
            should_reresearch=True,
            recommendations=["More recent data"],
            notes="First pass weak",
        ),
        TaskReview(
            coverage_score=0.5,
            reliability_score=0.5,
            clarity_score=0.5,
            overall_score=0.5,
            should_reresearch=True,
            recommendations=["Add case studies"],
            notes="Second pass still weak",
        ),
    ]
    agent.reviewer = _FakeReviewer(reviews)
    agent.summarizer = _FakeSummarizer(summary="FINAL2")

    state, task = _make_state_and_task()
    list(agent._execute_task(state, task, emit_stream=False, step=None))

    # We should have iterated at most twice due to the cap.
    assert len(calls) == 2
    # Combined sources should include the last SRC value we generated.
    assert "SRC2" in task.sources_summary
    assert task.status == TaskStatus.COMPLETED
    assert task.summary == "FINAL2"


def test_execute_task_streaming_event_sequence(monkeypatch: pytest.MonkeyPatch):
    agent = _setup_agent(monkeypatch, max_loops=1)

    def fake_dispatch(query, search_backend, fetch_full_page, loop_count):
        return (
            {"results": [{"title": "T", "url": "https://example.com", "content": "C"}]},
            [],
            None,
            "backend",
        )

    def fake_prepare(search_result, answer_text, fetch_full_page):
        return "SRC", "CTX"

    monkeypatch.setattr(agent_module, "dispatch_search", fake_dispatch)
    monkeypatch.setattr(agent_module, "prepare_research_context", fake_prepare)

    review = TaskReview(
        coverage_score=0.9,
        reliability_score=0.9,
        clarity_score=0.9,
        overall_score=0.9,
        should_reresearch=False,
        recommendations=[],
        notes="Fine",
    )
    agent.reviewer = _FakeReviewer([review])
    agent.summarizer = _FakeSummarizer(summary="IGNORED", stream_chunks=["a", "b"])

    state, task = _make_state_and_task()
    events = list(agent._execute_task(state, task, emit_stream=True, step=1))

    types = [e["type"] for e in events]
    # Expect sources -> review -> task_summary_chunk* -> task_status.
    assert types[0] == "sources"
    assert types[1] == "review"
    assert types[-1] == "task_status"
    # At least one summary chunk was streamed.
    assert "task_summary_chunk" in types
