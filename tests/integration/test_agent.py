"""Integration tests for DeepResearchAgent - calls real LLM + search."""

import pytest
from dotenv import load_dotenv

load_dotenv()

from backend.src.agent import DeepResearchAgent


@pytest.mark.integration
def test_agent_integration():
    """Integration test: end-to-end run() with real dependencies."""
    agent = DeepResearchAgent()
    result = agent.run("What is latest progress of the AI agent research?")

    # Minimal contract checks; content is non-deterministic.
    assert result.report_markdown is not None
    assert isinstance(result.todo_items, list)


@pytest.mark.integration
def test_agent_stream_integration():
    """Integration test: run_stream() with real dependencies."""
    agent = DeepResearchAgent()
    events = agent.run_stream("What is latest progress of the AI agent research?")

    # Ensure we eventually see a final_report and done event.
    types = []
    for event in events:
        types.append(event.get("type"))
    assert "final_report" in types
    assert "done" in types
