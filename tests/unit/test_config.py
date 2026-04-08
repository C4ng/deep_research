import pytest

from backend.src.config import Configuration, SearchAPI


def test_configuration_from_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Start from a clean env subset for relevant keys.
    for key in ["MAX_WEB_RESEARCH_LOOPS", "SEARCH_API", "FETCH_FULL_PAGE"]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("MAX_WEB_RESEARCH_LOOPS", "5")
    # Enum values are lower-case (see SearchAPI), so we use lower-case here.
    monkeypatch.setenv("SEARCH_API", "tavily")
    monkeypatch.setenv("FETCH_FULL_PAGE", "false")

    config = Configuration.from_env()

    assert config.max_web_research_loops == 5
    assert config.search_api == SearchAPI.TAVILY
    # Boolean fields should be parsed from strings.
    assert config.fetch_full_page is False
