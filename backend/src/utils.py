from __future__ import annotations
from typing import Any, Dict, List

import re

import logging
logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


def get_config_value(value: Any) -> str:
    """Return configuration value as plain string."""

    return value if isinstance(value, str) else value.value


def strip_thinking_tokens(text: str) -> str:
    """Remove ``<think>`` sections from model responses."""

    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text


def _clean_web_text(text: str) -> str:
    """Remove common scrape noise (images/data URIs/extra whitespace)."""
    if not text:
        return ""

    cleaned = text

    # Remove markdown images entirely: ![alt](url)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)
    # Remove raw data/image URIs that are often huge and useless.
    cleaned = re.sub(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+", " ", cleaned)
    # Drop very long base64-ish runs that still leak through.
    cleaned = re.sub(r"[A-Za-z0-9+/]{120,}={0,2}", " ", cleaned)

    # Collapse excessive whitespace while preserving line breaks reasonably.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def deduplicate_and_format_sources(
    search_response: Dict[str, Any] | List[Dict[str, Any]],
    max_tokens_per_source: int,
    *,
    fetch_full_page: bool = False,
) -> str:
    """Format and deduplicate search results for downstream prompting.
    
    Note: Results from search_tool are already deduplicated, but we deduplicate
    again here as a defensive measure in case results come from other sources.
    """

    if isinstance(search_response, dict):
        sources_list = search_response.get("results", [])
    else:
        sources_list = search_response

    # Deduplicate by URL (defensive - results should already be deduplicated by search_tool)
    unique_sources: dict[str, Dict[str, Any]] = {}
    for source in sources_list:
        url = source.get("url")
        if not url:
            continue
        if url not in unique_sources:
            unique_sources[url] = source

    formatted_parts: List[str] = []
    for source in unique_sources.values():
        title = source.get("title") or source.get("url", "")
        content = _clean_web_text(source.get("content", ""))
        formatted_parts.append(f"Source: {title}\n\n")
        formatted_parts.append(f"URL: {source.get('url', '')}\n\n")
        formatted_parts.append(f"Content: {content}\n\n")

        if fetch_full_page:
            raw_content = source.get("raw_content")
            if raw_content is None:
                logger.debug("raw_content missing for %s", source.get("url", ""))
                raw_content = ""
            raw_content = _clean_web_text(raw_content)
            char_limit = max_tokens_per_source * CHARS_PER_TOKEN
            if len(raw_content) > char_limit:
                raw_content = f"{raw_content[:char_limit]}... [truncated]"
            formatted_parts.append(
                f"Content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
            )

    return "".join(formatted_parts).strip()


def format_sources(search_results: Dict[str, Any] | None) -> str:
    """Return bullet list summarising search sources."""

    if not search_results:
        return ""

    results = search_results.get("results", [])
    return "\n".join(
        f"* {item.get('title', item.get('url', ''))} : {item.get('url', '')}"
        for item in results
        if item.get("url")
    )
