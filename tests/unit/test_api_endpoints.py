"""Unit tests for FastAPI endpoints using mocked agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.src.main import create_app
from backend.src.models import SummaryStateOutput, TodoItem


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestResearchEndpoint:
    @patch("backend.src.main.DeepResearchAgent")
    def test_success(self, mock_agent_cls, client):
        mock_agent = MagicMock()
        mock_agent.run.return_value = SummaryStateOutput(
            running_summary="summary",
            report_markdown="# Report",
            todo_items=[
                TodoItem(id=1, title="T1", intent="I1", query="Q1", summary="S1"),
            ],
        )
        mock_agent_cls.return_value = mock_agent

        resp = client.post("/research", json={"topic": "test"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["report_markdown"] == "# Report"
        assert len(body["todo_items"]) == 1

    @patch("backend.src.main._build_config", side_effect=ValueError("bad input"))
    def test_bad_request(self, _mock_cfg, client):
        resp = client.post("/research", json={"topic": "x"})
        assert resp.status_code == 400
        assert "bad input" in resp.json()["detail"]

    @patch("backend.src.main.DeepResearchAgent")
    def test_internal_error(self, mock_agent_cls, client):
        mock_agent_cls.return_value.run.side_effect = RuntimeError("boom")
        resp = client.post("/research", json={"topic": "x"})
        assert resp.status_code == 500

    @patch("backend.src.main.DeepResearchAgent")
    def test_pipeline_error(self, mock_agent_cls, client):
        from backend.src.exceptions import PlanningError

        mock_agent_cls.return_value.run.side_effect = PlanningError("plan failed")
        resp = client.post("/research", json={"topic": "x"})
        assert resp.status_code == 500
        assert "plan failed" in resp.json()["detail"]


class TestSSEEndpoint:
    @patch("backend.src.main.DeepResearchAgent")
    def test_stream_events(self, mock_agent_cls, client):
        def fake_stream(topic):
            yield {"type": "status", "message": "Starting"}
            yield {"type": "done"}

        mock_agent = MagicMock()
        mock_agent.run_stream.return_value = fake_stream("t")
        mock_agent_cls.return_value = mock_agent

        resp = client.post("/research/stream", json={"topic": "test"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = [
            line for line in resp.text.strip().split("\n\n") if line.startswith("data:")
        ]
        assert len(lines) == 2
        first = json.loads(lines[0].removeprefix("data: "))
        assert first["type"] == "status"
        last = json.loads(lines[1].removeprefix("data: "))
        assert last["type"] == "done"

    @patch("backend.src.main.DeepResearchAgent")
    def test_stream_error_event(self, mock_agent_cls, client):
        def failing_stream(topic):
            yield {"type": "status", "message": "Starting"}
            raise RuntimeError("stream failure")

        mock_agent = MagicMock()
        mock_agent.run_stream.return_value = failing_stream("t")
        mock_agent_cls.return_value = mock_agent

        resp = client.post("/research/stream", json={"topic": "test"})
        assert resp.status_code == 200
        lines = [
            line for line in resp.text.strip().split("\n\n") if line.startswith("data:")
        ]
        last_event = json.loads(lines[-1].removeprefix("data: "))
        assert last_event["type"] == "error"
        assert "stream failure" in last_event["detail"]
