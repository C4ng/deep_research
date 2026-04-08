from backend.src.models import SummaryState, TodoItem
from backend.src.services.summarizer import SummarizationService


class _FakeSummarizerAgent:
    def __init__(self, response: str):
        self._response = response
        self.last_prompt: str | None = None

    def run(self, prompt: str) -> str:  # pragma: no cover - trivial
        self.last_prompt = prompt
        return self._response

    def clear_history(self) -> None:  # pragma: no cover - trivial
        pass


def _make_state_and_task() -> tuple[SummaryState, TodoItem]:
    state = SummaryState(research_topic="Streaming LLMs")
    task = TodoItem(
        id=1,
        title="Task One",
        intent="Understand streaming behavior.",
        query="streaming llm behavior",
    )
    return state, task


def test_summarizer_build_prompt_includes_reviewer_notes():
    state, task = _make_state_and_task()
    task.review_notes = "Reviewer says: focus on reliability."
    context = "Some context."

    fake_agent = _FakeSummarizerAgent("ok")
    service = SummarizationService(lambda: fake_agent, strip_thinking_tokens=False)

    prompt = service._build_prompt(state, task, context)

    assert "Research topic: Streaming LLMs" in prompt
    assert "Task name: Task One" in prompt
    assert "Task intent: Understand streaming behavior." in prompt
    assert "Search query: streaming llm behavior" in prompt
    assert "Reviewer notes:" in prompt
    assert "Reviewer says: focus on reliability." in prompt
    assert "Task context:" in prompt
    assert "Some context." in prompt


def test_summarizer_strips_thinking_tokens_when_enabled():
    state, task = _make_state_and_task()
    context = "ctx"

    # Response contains <think> that should be stripped.
    fake_agent = _FakeSummarizerAgent("<think>hidden</think>Visible summary")
    service = SummarizationService(lambda: fake_agent, strip_thinking_tokens=True)

    summary = service.summarize_task(state, task, context)

    assert "hidden" not in summary
    assert "Visible summary" in summary
