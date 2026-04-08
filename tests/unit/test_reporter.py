from backend.src.models import SummaryState, TodoItem
from backend.src.services.reporter import ReportingService


class _FakeReportAgent:
    def __init__(self, response: str):
        self._response = response
        self.last_prompt: str | None = None

    def run(self, prompt: str) -> str:  # pragma: no cover - trivial
        self.last_prompt = prompt
        return self._response

    def clear_history(self) -> None:  # pragma: no cover - trivial
        pass


def test_reporting_prompt_and_strip_think():
    state = SummaryState(research_topic="Test topic")
    state.todo_items = [
        TodoItem(
            id=1,
            title="T1",
            intent="I1",
            query="Q1",
            summary="Summary 1",
            sources_summary="Sources 1",
        ),
        TodoItem(
            id=2,
            title="T2",
            intent="I2",
            query="Q2",
            summary="Summary 2",
            sources_summary="Sources 2",
        ),
    ]

    fake_agent = _FakeReportAgent("<think>internal</think>Final report body")
    service = ReportingService(fake_agent, strip_thinking_tokens=True)

    report = service.generate_report(state)

    # Report text should have stripped <think>...</think>.
    assert "internal" not in report
    assert "Final report body" in report

    # Prompt sent to the agent should include both tasks and their fields.
    assert fake_agent.last_prompt is not None
    prompt = fake_agent.last_prompt
    assert "Research topic: Test topic" in prompt
    assert "Task 1: T1" in prompt
    assert "Task 2: T2" in prompt
    assert "Task intent: I1" in prompt
    assert "Task intent: I2" in prompt
    assert "Search query: Q1" in prompt
    assert "Search query: Q2" in prompt
    assert "Execution status:" in prompt
    assert "Task summary:" in prompt
    assert "Sources summary:" in prompt

