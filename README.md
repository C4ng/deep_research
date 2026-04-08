# Deep Research Agent

An agentic deep research assistant that decomposes complex topics into focused tasks, iteratively searches the web, reflects on evidence quality, and produces a structured Markdown report.

## Architecture

```
User  ─── POST /research/stream ───▶  FastAPI (SSE)
                                          │
                                   DeepResearchAgent
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
              PlanningService      Per-Task Loop         ReportingService
              (decompose topic     (for each task)       (combine summaries
               into tasks)              │                 into final report)
                                        │
                           ┌────────────┼────────────┐
                           ▼            ▼            ▼
                     SearchService  Summarizer   ReviewerService
                     (Tavily web    (draft       (rubric scoring:
                      search)       findings)    coverage, reliability,
                                                 clarity, overall)
                                        │
                                        ▼
                                  Score OK? ──No──▶ Refine query → re-search
                                    │
                                   Yes
                                    │
                                    ▼
                              Final Summary
```

### Reflection Mechanism

The reviewer acts as a feedback controller between search and summarization:

1. After each search pass, the summarizer creates a draft findings summary.
2. The reviewer scores it on four dimensions (coverage, reliability, clarity, overall) using a structured JSON rubric.
3. If `should_reresearch=true` and the loop cap hasn't been reached, the agent refines its query using reviewer recommendations and runs another search pass.
4. This creates a bounded iterative improvement loop (max 2 iterations per task) that improves factual robustness without unbounded cost.

### Caching Strategy

Two-layer cache to avoid redundant LLM/search calls:

- **In-memory LLM cache** (`LLMCache`): keyed by `(namespace, sha256(prompt))`, used across planner, reviewer, summarizer, and reporter within a single process.
- **Filesystem stage cache** (`StageFileCache`): persists intermediate results (planner, search, review, summary, report) per topic/task across restarts. Controlled via `DEEP_RESEARCH_FILE_CACHE` env var.

## Project Structure

```
deep_research/
├── agent/                      # LLM client and agent abstractions
│   └── src/
│       ├── llm.py              # OpenAI-compatible client with retry logic
│       ├── agent.py            # Base Agent ABC
│       ├── message.py          # Chat message model
│       ├── agents/
│       │   └── simple_agent.py # Non-streaming + streaming agent
│       └── tools/
│           └── builtin/
│               └── search_tools.py  # Tavily search tool wrapper
│
├── backend/                    # Research pipeline and API
│   └── src/
│       ├── main.py             # FastAPI app (REST + SSE endpoints)
│       ├── agent.py            # DeepResearchAgent orchestrator
│       ├── config.py           # Pydantic configuration from env
│       ├── models.py           # Domain models (TodoItem, TaskReview, etc.)
│       ├── prompts.py          # System prompts for each agent role
│       ├── cache.py            # LLMCache + StageFileCache
│       ├── utils.py            # Text processing utilities
│       ├── exceptions.py       # Backend exception hierarchy
│       └── services/
│           ├── planner.py      # Task decomposition
│           ├── search.py       # Web search dispatch
│           ├── reviewer.py     # Rubric-based reflection
│           ├── summarizer.py   # Task summarization
│           └── reporter.py     # Final report generation
│
├── tests/
│   ├── conftest.py             # Shared fixtures and test config
│   ├── unit/                   # Fast, deterministic tests (no network)
│   └── integration/            # Tests hitting real LLM/search backends
│
├── docs/
│   └── STREAMING_CLIENT.md     # SSE event types and client examples
│
├── pyproject.toml              # Project metadata, pytest, ruff, mypy config
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Dev/test dependencies (includes runtime)
├── Makefile
├── .github/workflows/ci.yml   # Lint + format + typecheck + unit tests
├── .env.example                # Required environment variables template
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.11+
- API keys for LLM provider and Tavily search

### Installation

```bash
# Clone and install
git clone <repo-url>
cd deep_research

# Install runtime dependencies
pip install -r requirements.txt

# Or install with dev/test dependencies
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Running

```bash
# Start the API server
make serve

# Or directly
uvicorn backend.src.main:app --host 0.0.0.0 --port 8000 --reload
```

## API

### `GET /healthz`

Health check endpoint.

### `POST /research`

Synchronous research endpoint. Returns a complete report.

```json
{
  "topic": "What is the latest progress of AI agent research?"
}
```

### `POST /research/stream`

Server-Sent Events endpoint streaming progress in real-time. See [docs/STREAMING_CLIENT.md](docs/STREAMING_CLIENT.md) for event types and client examples.

## Testing

```bash
# Run unit tests (default, no API keys needed)
make test-unit

# Run integration tests (requires .env with valid keys)
make test-integration

# Lint and format
make lint
make format

# Type check
make typecheck
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Reviewer uses strict JSON output | Machine-parseable rubric scores drive control flow; Markdown would require fragile extraction |
| Summarizer/Reporter use Markdown | User-facing narrative output benefits from rich formatting |
| Max 2 search iterations per task | Bounds latency and cost while still allowing targeted quality improvement |
| Max 2 tasks per topic | Keeps research focused; prevents scope explosion on broad topics |
| Dual cache (memory + filesystem) | In-memory for within-session dedup; filesystem for cross-restart persistence |
| Reviewer evaluates summarized findings | Avoids feeding verbose raw search context into reviewer, reducing truncation risk |
| Tenacity retry with separate policies | Rate-limit errors need longer backoff than transient server errors |
