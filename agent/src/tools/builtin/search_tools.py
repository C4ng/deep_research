import os
import importlib.util
from pydantic import BaseModel, Field
from typing import Literal, Any, Dict, Iterable
from abc import ABC, abstractmethod


from agent.src.tools.base import Tool

from tavily import TavilyClient 

import logging
logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
SUPPORTED_BACKENDS = Literal[
    "tavily",
]
SUPPORTED_RETURN_MODES = Literal[
    "text",
    "structured",
]


def _limit_text(text: str, token_limit: int) -> str:
    char_limit = token_limit * CHARS_PER_TOKEN
    if len(text) <= char_limit:
        return text
    return text[:char_limit] + "... [truncated]"

def _normalized_result(
    *,
    title: str,
    url: str,
    content: str,
    raw_content: str | None,
) -> Dict[str, str]:
    payload: Dict[str, str] = {
        "title": title or url,
        "url": url,
        "content": content or "",
    }
    if raw_content is not None:
        payload["raw_content"] = raw_content
    return payload


def _structured_payload(
    results: Iterable[Dict[str, Any]],
    *,
    backend: str,
    answer: str | None = None,
    notices: Iterable[str] | None = None,
) -> Dict[str, Any]:
    return {
        "results": list(results),
        "backend": backend,
        "answer": answer,
        "notices": list(notices or []),
    }

class SearchProvider(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        """Check if library is installed and API keys are present."""
        pass

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[dict]:
        pass

class TavilyProvider(SearchProvider):
    def __init__(self):
        self._client = None 

    def is_available(self) -> bool:
        has_lib = importlib.util.find_spec("tavily") is not None
        has_key = bool(os.getenv("TAVILY_API_KEY"))
        if not has_lib:
            logger.warning("Tavily library not found")
        if not has_key:
            logger.warning("TavILY_API_KEY not found")
        return has_lib and has_key

    def _get_client(self):
        """Internal helper to ensure the client is created only once."""
        if self._client is None:
            api_key = os.getenv("TAVILY_API_KEY")
            self._client = TavilyClient(api_key=api_key)
            logger.info("Tavily client initialized and cached.")
        return self._client

    def search(
            self, 
            query: str, 
            fetch_full_page: bool, 
            max_results: int, 
            max_tokens: int, 
            loop_count: int
        ) -> list[dict]:

        response = self._get_client().search(  # type: ignore[call-arg]
            query=query,
            max_results=max_results,
            include_raw_content=fetch_full_page,
            include_answer=True, 
        )

        results = []
        for item in response.get("results", [])[:max_results]:
            raw_content = None
            if fetch_full_page:
                raw_content = item.get("raw_content")
                if raw_content:
                    raw_content = _limit_text(raw_content, max_tokens)
            results.append(
                _normalized_result(
                    title=item.get("title") or item.get("url", ""),
                    url=item.get("url", ""),
                    content=item.get("content") or "",
                    raw_content=raw_content,
                )
            )

        return _structured_payload(
            results,
            backend="tavily",
            answer=response.get("answer"),
        )


class SearchSchema(BaseModel):
    query: str = Field(..., description="The search query keywords")
    backend: SUPPORTED_BACKENDS = Field(
        default="tavily", 
        description="The search engine to use"
    )
    return_mode: SUPPORTED_RETURN_MODES = Field(default="text", description="The mode to return the results in")
    max_results: int = Field(default=5, ge=1, le=10, description="Number of results to return")
    fetch_full_page: bool = Field(default=False, description="Whether to scrape the full content of the pages")
    max_tokens: int = Field(default=2000, ge=100, le=4000, description="Max tokens per source")
    loop_count: int = Field(default=0, ge=0, le=3, description="Current research loop iteration (0-indexed)")


class SearchTool(Tool):
    """Search Tool supporting Tavily."""

    @property
    def args_schema(self):
        return SearchSchema

    def __init__(self) -> None:
        super().__init__(
            name="search",
            description=f"Search the web."
        )
        self.providers = {
            "tavily": TavilyProvider(),
        }
     
    def _execute(
        self, 
        query: str, 
        backend: SUPPORTED_BACKENDS, 
        return_mode: SUPPORTED_RETURN_MODES,
        max_results: int, 
        fetch_full_page: bool,
        max_tokens: int,
        loop_count: int,
    ) -> str:
        """
        The core logic. 
        kwargs are now automatically mapped to these named arguments!
        """

        results = []
        
        provider = self.providers.get(backend)
        if provider and provider.is_available():
            results = provider.search(
                query=query,
                fetch_full_page=fetch_full_page, 
                max_results=max_results,
                max_tokens=max_tokens,
                loop_count=loop_count)
        else:
            # Tell the LLM why it failed so it can try another way
            return f"Error: Backend '{backend}' is currently unavailable (check API keys or installation)."

        if not results:
            return "No relevant search results found. Try broader keywords."

        if return_mode == "structured":
            return results

        # 3. Format the final output for the LLM
        return self._format_text_response(query=query, payload=results)
    
    def _format_text_response(self, *, query: str, payload: Dict[str, Any]) -> str:
        answer = payload.get("answer")
        notices = payload.get("notices") or []
        results = payload.get("results") or []
        backend = payload.get("backend")

        lines = [f"Search query: {query}", f"Using search backend: {backend}"]
        if answer:
            lines.append(f"Direct answer: {answer}")

        if results:
            lines.append("")
            lines.append("References:")
            for idx, item in enumerate(results, start=1):
                title = item.get("title") or item.get("url", "")
                lines.append(f"[{idx}] {title}")
                if item.get("content"):
                    lines.append(f"    {item['content']}")
                if item.get("url"):
                    lines.append(f"    Source: {item['url']}")
                lines.append("")
        else:
            lines.append("No related search results found.")

        if notices:
            lines.append("Notices:")
            for notice in notices:
                if notice:
                    lines.append(f"- {notice}")

        return "\n".join(line for line in lines if line is not None)