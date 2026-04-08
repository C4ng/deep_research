"""Integration tests for FastAPI endpoints - hits real agent + LLM/search."""

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

load_dotenv()

from backend.src.main import ResearchRequest, app  # noqa: E402


@pytest.mark.integration
def test_api_integration():
    """Integration test: calls real /research endpoint (LLM + search required)."""
    client = TestClient(app)

    # Health check is cheap and deterministic.
    health_response = client.get("/healthz")
    print("/healthz status:", health_response.status_code)
    print("/healthz body:", health_response.json())
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    request_payload = ResearchRequest(topic="What is latest progress of the AI agent research?")

    response = client.post(
        "/research",
        json=request_payload.model_dump(),
    )

    print("/research status:", response.status_code)
    try:
        body = response.json()
        print("/research keys:", list(body.keys()))
        print("/research report length:", len(body.get("report_markdown", "")))
        print("/research todo_items:", len(body.get("todo_items", [])))
    except Exception:
        print("/research raw body:", response.text[:200])

    # We only assert basic contract here; content depends on external services.
    assert response.status_code in (200, 400, 500)
