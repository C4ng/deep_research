import pytest
from pydantic import ValidationError

from backend.src.models import SummaryState, TaskReview, TodoItem
from backend.src.services.reviewer import ReviewerService


class _FakeAgent:
    """Minimal stand-in for SimpleAgent used by ReviewerService."""

    def __init__(self, response: str):
        self._response = response

    def run(self, prompt: str) -> str:  # pragma: no cover - trivial
        # We ignore the prompt in tests and just return the canned response.
        if isinstance(self._response, list):
            if self._response:
                return self._response.pop(0)
            return ""
        return self._response

    def clear_history(self) -> None:  # pragma: no cover - trivial
        pass


def _make_state_and_task() -> tuple[SummaryState, TodoItem]:
    state = SummaryState(research_topic="Test topic")
    task = TodoItem(
        id=1,
        title="Test Task",
        intent="Understand the test behavior.",
        query="test query",
    )
    return state, task


def test_reviewer_parses_valid_json_payload():
    state, task = _make_state_and_task()
    payload = {
        "coverage_score": 0.8,
        "reliability_score": 0.9,
        "clarity_score": 0.7,
        "overall_score": 0.75,
        "should_reresearch": False,
        "recommendations": ["Looks good"],
        "notes": "Solid coverage with minor gaps.",
    }

    import json

    response = json.dumps(payload)
    reviewer = ReviewerService(_FakeAgent(response), strip_thinking_tokens=False)

    review = reviewer.review_task(state, task, context="some context")

    assert isinstance(review, TaskReview)
    assert review.coverage_score == pytest.approx(0.8)
    assert review.reliability_score == pytest.approx(0.9)
    assert review.clarity_score == pytest.approx(0.7)
    assert review.overall_score == pytest.approx(0.75)
    assert review.should_reresearch is False
    assert review.recommendations == ["Looks good"]
    assert "Solid coverage" in review.notes


def test_reviewer_extracts_json_from_wrapped_output():
    state, task = _make_state_and_task()

    wrapped = """Some leading text
<think>internal chain of thought</think>
{
  "coverage_score": 0.6,
  "reliability_score": 0.5,
  "clarity_score": 0.4,
  "overall_score": 0.5,
  "should_reresearch": true,
  "recommendations": ["Add more recent sources"],
  "notes": "Needs more up-to-date evidence."
}
Trailing commentary that should be ignored."""

    reviewer = ReviewerService(_FakeAgent(wrapped), strip_thinking_tokens=True)

    review = reviewer.review_task(state, task, context="some context")

    # JSON inside the wrapper should be parsed correctly.
    assert isinstance(review, TaskReview)
    assert review.should_reresearch is True
    assert review.overall_score == pytest.approx(0.5)
    assert review.recommendations == ["Add more recent sources"]
    assert "Needs more up-to-date evidence." in review.notes


def test_reviewer_falls_back_on_invalid_output():
    state, task = _make_state_and_task()

    # No JSON object at all – should trigger fallback.
    bad_output = "This is not JSON at all."
    reviewer = ReviewerService(_FakeAgent(bad_output), strip_thinking_tokens=False)

    review = reviewer.review_task(state, task, context="some context")

    # Fallback review should be fully positive and not request re-research.
    assert isinstance(review, TaskReview)
    assert review.coverage_score == pytest.approx(1.0)
    assert review.reliability_score == pytest.approx(1.0)
    assert review.clarity_score == pytest.approx(1.0)
    assert review.overall_score == pytest.approx(1.0)
    assert review.should_reresearch is False
    assert "This is not JSON at all." in review.notes


def test_task_review_rejects_out_of_range_scores():
    # Any score outside [0, 1] should raise a ValidationError.
    with pytest.raises(ValidationError):
        TaskReview(
            coverage_score=1.2,
            reliability_score=0.5,
            clarity_score=0.5,
            overall_score=0.6,
            should_reresearch=False,
            recommendations=[],
            notes="Invalid coverage score",
        )


def test_reviewer_parses_markdown_fenced_json():
    state, task = _make_state_and_task()
    fenced = """```json
{
  "coverage_score": 0.7,
  "reliability_score": 0.8,
  "clarity_score": 0.9,
  "overall_score": 0.8,
  "should_reresearch": false,
  "recommendations": ["Keep concise"],
  "notes": "Fenced JSON should parse."
}
```"""

    reviewer = ReviewerService(_FakeAgent(fenced), strip_thinking_tokens=True)
    review = reviewer.review_task(state, task, context="ctx")
    assert review.overall_score == pytest.approx(0.8)
    assert review.should_reresearch is False


def test_reviewer_partial_parse_recovers_truncated_json():
    """When the JSON is truncated but contains recognisable fields, partial parsing succeeds."""
    state, task = _make_state_and_task()
    truncated = '```json\n{ "coverage_score": 0.9'
    reviewer = ReviewerService(_FakeAgent(truncated), strip_thinking_tokens=True)
    review = reviewer.review_task(state, task, context="ctx")
    assert review.coverage_score == pytest.approx(0.9)
    # Unrecoverable fields fall back to defaults.
    assert review.reliability_score == pytest.approx(0.5)
    assert review.clarity_score == pytest.approx(0.5)
