from backend.src.config import Configuration
from backend.src.models import SummaryState, TodoItem
from backend.src.services.planner import PlanningService


class _FakePlannerAgent:
    def __init__(self, response: str):
        self._response = response

    def run(self, prompt: str) -> str:  # pragma: no cover - trivial
        return self._response

    def clear_history(self) -> None:  # pragma: no cover - trivial
        pass


def _make_state() -> SummaryState:
    return SummaryState(research_topic="AI agents")


def test_planning_extract_json_payload_object_and_array():
    config = Configuration()
    service = PlanningService(_FakePlannerAgent("{}"), config)

    # Object with tasks
    text_obj = """
    Some preface...
    {
      "tasks": [
        {"title": "T1", "intent": "I1", "query": "Q1"}
      ]
    }
    trailing...
    """
    payload_obj = service._extract_json_payload(text_obj)
    assert isinstance(payload_obj, dict)
    assert payload_obj["tasks"][0]["title"] == "T1"

    # Pure array
    text_arr = """
    [
      {"title": "A1", "intent": "I1", "query": "Q1"},
      {"title": "A2", "intent": "I2", "query": "Q2"}
    ]
    """
    payload_arr = service._extract_json_payload(text_arr)
    # Implementation prefers extracting the first JSON object; we just ensure it is parsed.
    assert isinstance(payload_arr, (list, dict))
    if isinstance(payload_arr, list):
        assert payload_arr[1]["title"] == "A2"
    else:
        assert payload_arr["title"] == "A1"


def test_planning_extract_tasks_and_fallback():
    config = Configuration(strip_thinking_tokens=True)
    service = PlanningService(_FakePlannerAgent(""), config)

    # Valid JSON with tasks
    raw = """
    <think>internal</think>
    {
      "tasks": [
        {"title": "Task A", "intent": "Intent A", "query": "Query A"},
        {"title": "Task B", "intent": "Intent B", "query": "Query B"}
      ]
    }
    """
    tasks = service._extract_tasks(raw)
    assert len(tasks) == 2
    assert tasks[0]["title"] == "Task A"

    # When tasks payload is empty, plan_todo_list should create a fallback task.
    empty_payload = '{"tasks": []}'
    fallback_service = PlanningService(_FakePlannerAgent(empty_payload), config)
    state = _make_state()

    todo_items = fallback_service.plan_todo_list(state)
    assert len(todo_items) == 1
    task = todo_items[0]
    assert isinstance(task, TodoItem)
    assert "Background Research" in task.title
    assert "AI agents" in task.query
