"""Simple integration test for SearchTool - hits real backend."""

import pytest
from dotenv import load_dotenv

load_dotenv()

from agent.src.tools.builtin.search_tools import SearchTool


@pytest.mark.integration
def test_search_integration():
    """Integration test: calls real SearchTool backend (network + API key required)."""
    tool = SearchTool()

    result = tool.run(
        {
            "query": "What is Python programming?",
            "backend": "tavily",
            "max_results": 3,
        }
    )

    print("SearchTool result type:", type(result))
    if isinstance(result, dict):
        print("SearchTool keys:", list(result.keys()))
        print("SearchTool results count:", len(result.get("results", [])))
    else:
        print("SearchTool raw result (truncated):", str(result)[:200])

    # Basic sanity checks on structure; details depend on remote backend.
    assert isinstance(result, (dict, str))
