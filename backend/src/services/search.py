"""Search dispatch helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from backend.src.exceptions import SearchError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

from agent.src.tools.builtin.search_tools import SearchTool, SUPPORTED_BACKENDS

from backend.src.utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
)

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_SOURCE = 2000
_GLOBAL_SEARCH_TOOL: SearchTool | None = None


def _get_search_tool() -> SearchTool:
    """Get or create the global search tool instance (lazy initialization)."""
    global _GLOBAL_SEARCH_TOOL
    if _GLOBAL_SEARCH_TOOL is None:
        _GLOBAL_SEARCH_TOOL = SearchTool()
    return _GLOBAL_SEARCH_TOOL


def dispatch_search(
    query: str,
    search_backend: SUPPORTED_BACKENDS,
    fetch_full_page: bool,
    loop_count: int,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str]:
    """Execute configured search backend and normalise response payload."""

    # Convert enum to string if needed (SearchAPI enum -> "tavily" string)
    backend_str = get_config_value(search_backend)
   
    try:
        search_tool = _get_search_tool()
        raw_response = search_tool.run(
            {
                "query": query,
                "backend": backend_str,
                "return_mode": "structured",
                "max_results": 5,
                "fetch_full_page": fetch_full_page,
                "max_tokens": MAX_TOKENS_PER_SOURCE,
                "loop_count": loop_count,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        raise SearchError(f"Search backend {search_backend} failed: {exc}") from exc

    if isinstance(raw_response, str):
        notices = [raw_response]
        logger.warning("Search backend %s returned text notice: %s", search_backend, raw_response)
        payload: dict[str, Any] = {
            "results": [],
            "backend": backend_str,
            "answer": None,
            "notices": notices,
        }
    else:
        payload = raw_response
        notices = list(payload.get("notices") or [])
        # Ensure backend is a string, not enum
        if "backend" in payload:
            payload["backend"] = str(payload["backend"])

    backend_label = str(payload.get("backend") or backend_str)
    answer_text = payload.get("answer")
    results = payload.get("results", [])

    if notices:
        for notice in notices:
            logger.info("Search notice (%s): %s", backend_label, notice)

    logger.info(
        "Search backend=%s resolved_backend=%s answer=%s results=%s",
        search_backend,
        backend_label,
        bool(answer_text),
        len(results),
    )

    return payload, notices, answer_text, backend_label


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: Optional[str],
    fetch_full_page: bool,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""

    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=fetch_full_page,
    )

    if answer_text:
        context = f"Direct Answer:\n{answer_text}\n\n{context}"

    return sources_summary, context
