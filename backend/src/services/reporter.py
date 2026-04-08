"""Service that consolidates task results into the final report."""

from __future__ import annotations

import logging

from agent.src.agents.simple_agent import SimpleAgent
from backend.src.cache import llm_cache
from backend.src.exceptions import ReportError
from backend.src.models import SummaryState
from backend.src.utils import strip_thinking_tokens

logger = logging.getLogger(__name__)


class ReportingService:
    """Generates the final structured report."""

    def __init__(self, report_agent: SimpleAgent, strip_thinking_tokens: bool) -> None:
        self._agent = report_agent
        self._strip_thinking_tokens = strip_thinking_tokens

    def generate_report(self, state: SummaryState) -> str:
        """Generate a structured report based on completed tasks."""

        tasks_block = []
        for task in state.todo_items:
            summary_block = task.summary or "No available information"
            sources_block = task.sources_summary or "No available sources"
            tasks_block.append(
                f"### Task {task.id}: {task.title}\n"
                f"- Task intent: {task.intent}\n"
                f"- Search query: {task.query}\n"
                f"- Execution status: {task.status}\n"
                f"- Task summary: \n{summary_block}\n"
                f"- Sources summary: \n{sources_block}\n"
            )

        prompt = f"Research topic: {state.research_topic}\nTask overview: \n{''.join(tasks_block)}\n"

        logger.debug("Reporter prompt (truncated): %s", prompt[:1000])

        cached = llm_cache.get("reporter", prompt)
        if cached is not None:
            report_text = cached
            logger.info("Reporter cache hit for topic %r", state.research_topic)
        else:
            try:
                response = self._agent.run(prompt)
            except Exception as e:
                raise ReportError(f"Report generation LLM call failed: {e}") from e
            finally:
                self._agent.clear_history()
            report_text = (response or "").strip()
        if self._strip_thinking_tokens:
            report_text = strip_thinking_tokens(report_text)

        logger.info("Reporter raw output (truncated): %s", report_text[:1000])

        if report_text:
            llm_cache.set("reporter", prompt, report_text)

        return report_text or "Report generation failed, please check the input."
