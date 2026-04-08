import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.src.agent import DeepResearchAgent
from backend.src.config import Configuration, SearchAPI
from backend.src.exceptions import DeepResearchError

logger = logging.getLogger(__name__)

_logging_initialized = False


def _setup_logging() -> None:
    """Configure console + file logging once. Safe to call multiple times.

    The file handler is skipped when ``TESTING=1`` is set (e.g. by pytest)
    to avoid polluting the workspace with log files during test runs.
    """
    import os

    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    if os.getenv("TESTING") != "1":
        file_handler = logging.FileHandler("backend_debug.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

class ResearchRequest(BaseModel):
    """Payload for triggering a research run."""

    topic: str = Field(..., description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )

class ResearchResponse(BaseModel):
    """HTTP response containing the generated report and structured tasks."""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Aggregate LLM token consumption for the pipeline run",
    )

def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"

def _build_config(payload: ResearchRequest) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api

    return Configuration.from_env(overrides=overrides)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    # Startup
    config = Configuration.from_env()
    base_url = config.llm_base_url or "unset"

    logger.info(
        "DeepResearch configuration loaded: model=%s base_url=%s search_api=%s "
        "max_loops=%s fetch_full_page=%s strip_thinking=%s api_key=%s",
        config.llm_model_id,
        base_url,
        config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api,
        config.max_web_research_loops,
        config.fetch_full_page,
        config.strip_thinking_tokens,
        _mask_secret(config.llm_api_key),
    )
    
    yield
    
    # Shutdown (if needed in the future)
    logger.info("Application shutting down...")

def create_app() -> FastAPI:
    _setup_logging()
    app = FastAPI(title="Deep Research Agent API", lifespan=lifespan)

    cors_config = Configuration.from_env()
    origins = [o.strip() for o in cors_config.cors_allowed_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}
    
    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest) -> ResearchResponse:
        try:
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
            result = agent.run(payload.topic)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DeepResearchError as exc:
            logger.error("Research pipeline error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected error during research")
            raise HTTPException(status_code=500, detail="Internal server error") from exc

        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary
            }
            for item in result.todo_items
        ]

        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
            usage=result.usage,
        )

    @app.post("/research/stream")
    def run_research_stream(payload: ResearchRequest) -> StreamingResponse:
        """
        Run research and stream progress via Server-Sent Events (SSE).
        Each event is a JSON object in the form: data: {...}\\n\\n
        See docs/STREAMING_CLIENT.md for event types and client usage.
        """
        try:
            config = _build_config(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_stream():
            agent = DeepResearchAgent(config=config)
            try:
                for event in agent.run_stream(payload.topic):
                    # SSE format: "data: " + JSON + "\\n\\n"
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                logger.exception("Research stream failed: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
