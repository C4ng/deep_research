"""Task summarization utilities."""

from __future__ import annotations

from collections.abc import Callable

from agent.src.agents.simple_agent import SimpleAgent

from backend.src.models import SummaryState, TodoItem
from backend.src.utils import strip_thinking_tokens
from backend.src.cache import llm_cache
from backend.src.exceptions import SummarizationError

from typing import Tuple, Iterator

import logging
logger = logging.getLogger(__name__)


class SummarizationService:
    """Handles synchronous and streaming task summarization."""

    def __init__(
        self,
        summarizer_factory: Callable[[], SimpleAgent],
        strip_thinking_tokens: bool,
    ) -> None:
        self._agent_factory = summarizer_factory
        self._strip_thinking_tokens = strip_thinking_tokens

    def summarize_task(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """Generate a task-specific summary using the summarizer agent."""

        prompt = self._build_prompt(state, task, context)

        logger.debug(
            "Summarizer prompt for task %d (truncated): %s",
            task.id,
            prompt[:1000],
        )

        cached = llm_cache.get("summarizer", prompt)
        if cached is not None:
            summary_text = cached
            logger.info("Summarizer cache hit for task %d", task.id)
        else:
            agent = self._agent_factory()
            try:
                response = agent.run(prompt)
            except Exception as e:
                raise SummarizationError(f"Summarization failed for task {task.id}: {e}") from e
            finally:
                agent.clear_history()

            summary_text = (response or "").strip()
        if self._strip_thinking_tokens:
            summary_text = strip_thinking_tokens(summary_text)

        logger.info(
            "Summarizer raw output for task %d (truncated): %s",
            task.id,
            summary_text[:1000],
        )

        if summary_text:
            llm_cache.set("summarizer", prompt, summary_text)

        return summary_text or "No available information"
    
    def stream_task_summary(
        self, state: SummaryState, task: TodoItem, context: str
        ) -> Tuple[Iterator[str], Callable[[], str]]:
            prompt = self._build_prompt(state, task, context)
            remove_thinking = self._strip_thinking_tokens
            cached = llm_cache.get("summarizer", prompt)

            # If we already have a cached summary, stream from cache instead of calling LLM.
            if cached is not None:
                def cached_gen() -> Iterator[str]:
                    if cached:
                        yield cached

                def cached_get() -> str:
                    return cached

                return cached_gen(), cached_get

            raw_buffer = ""
            visible_output = ""
            emit_index = 0
            agent = self._agent_factory()
        
            def flush_visible() -> Iterator[str]:
                nonlocal emit_index, raw_buffer
                while True:
                    start = raw_buffer.find("<think>", emit_index)
                    if start == -1:
                        if emit_index < len(raw_buffer):
                            segment = raw_buffer[emit_index:]
                            emit_index = len(raw_buffer)
                            if segment:
                                yield segment
                        break

                    if start > emit_index:
                        segment = raw_buffer[emit_index:start]
                        emit_index = start
                        if segment:
                            yield segment

                    end = raw_buffer.find("</think>", start)
                    if end == -1:
                        break
                    emit_index = end + len("</think>")

            def generator() -> Iterator[str]:
                nonlocal raw_buffer, visible_output, emit_index
                try:
                    for chunk in agent.stream_run(prompt):
                        raw_buffer += chunk
                        if remove_thinking:
                            for segment in flush_visible():
                                visible_output += segment
                                if segment:
                                    yield segment
                        else:
                            visible_output += chunk
                            if chunk:
                                yield chunk
                finally:
                    if remove_thinking:
                        for segment in flush_visible():
                            visible_output += segment
                            if segment:
                                yield segment
                    agent.clear_history()
                
            
            def get_summary() -> str:
                if remove_thinking:
                    cleaned = strip_thinking_tokens(visible_output)
                else:
                    cleaned = visible_output
                return cleaned
            
            return generator(), get_summary

   
    def _build_prompt(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """Construct the summarization prompt shared by both modes."""

        review_block = task.review_notes or "No reviewer notes available."

        return (
            f"Research topic: {state.research_topic}\n"
            f"Task name: {task.title}\n"
            f"Task intent: {task.intent}\n"
            f"Search query: {task.query}\n"
            f"Reviewer notes: \n{review_block}\n"
            f"Task context: \n{context}\n"
            "Please return a Markdown summary for the user based on the above context. (Still follow the task summary template.)"
        )
