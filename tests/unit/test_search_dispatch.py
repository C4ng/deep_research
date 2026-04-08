import pytest

from backend.src.config import Configuration
import backend.src.services.search as search_module


class _FakeSearchTool:
    def __init__(self, response):
        self._response = response
        self.last_payload = None

    def run(self, payload: dict) -> dict | str:  # pragma: no cover - trivial
        self.last_payload = payload
        return self._response


def test_dispatch_search_structured_response(monkeypatch: pytest.MonkeyPatch) -> None:
    response = {
        "results": [{"title": "T", "url": "https://example.com", "content": "C"}],
        "backend": "tavily",
        "answer": "Ans",
        "notices": ["ok"],
    }
    fake_tool = _FakeSearchTool(response)
    monkeypatch.setattr(search_module, "_get_search_tool", lambda: fake_tool)

    config = Configuration()
    payload, notices, answer, backend = search_module.dispatch_search(
        query="q",
        search_backend=config.search_api,
        fetch_full_page=False,
        loop_count=1,
    )

    assert payload["results"] == response["results"]
    assert notices == ["ok"]
    assert answer == "Ans"
    assert backend == "tavily"
    # The payload given to the search tool should include our query and backend string.
    assert fake_tool.last_payload["query"] == "q"
    assert fake_tool.last_payload["backend"] == "tavily"


def test_dispatch_search_text_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    # When the tool returns a plain string, dispatch_search should wrap it.
    fake_tool = _FakeSearchTool("temporary error")
    monkeypatch.setattr(search_module, "_get_search_tool", lambda: fake_tool)

    config = Configuration()
    payload, notices, answer, backend = search_module.dispatch_search(
        query="q2",
        search_backend=config.search_api,
        fetch_full_page=True,
        loop_count=2,
    )

    assert notices == ["temporary error"]
    assert payload["results"] == []
    assert payload["backend"] == backend
    assert answer is None

