from __future__ import annotations

import json
import logging
from typing import Any

from agent.src.agents.simple_agent import SimpleAgent
from backend.src.cache import llm_cache
from backend.src.config import Configuration
from backend.src.exceptions import PlanningError
from backend.src.models import SummaryState, TodoItem
from backend.src.prompts import get_current_date, todo_planner_instructions
from backend.src.utils import strip_thinking_tokens

logger = logging.getLogger(__name__)


class PlanningService:
    """Wraps the planner agent to produce structured TODO items."""

    def __init__(self, planner_agent: SimpleAgent, config: Configuration) -> None:
        self._agent = planner_agent
        self._strip_thinking_tokens = config.strip_thinking_tokens

    def plan_todo_list(self, state: SummaryState) -> list[TodoItem]:
        """Ask the planner agent to break the topic into actionable tasks."""

        prompt = todo_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
        )

        # Debug logging for prompt (truncated to avoid huge logs).
        logger.debug("Planner prompt (truncated): %s", prompt[:1000])

        # Try cache first to avoid repeated planner calls.
        cached = llm_cache.get("planner", prompt)
        if cached is not None:
            response = cached
            logger.info("Planner cache hit for topic %r", state.research_topic)
        else:
            response = ""
            try:
                response = self._agent.run(prompt)
                if response:
                    llm_cache.set("planner", prompt, response)
            except Exception as e:
                raise PlanningError(f"Planner LLM call failed: {e}") from e
            finally:
                self._agent.clear_history()

        logger.info("Planner raw output (truncated): %s", response[:1000] if response else "")

        tasks_payload = self._extract_tasks(response)
        todo_items: list[TodoItem] = []

        if not tasks_payload:
            return [self.create_fallback_task(state)]

        for idx, item in enumerate(tasks_payload, start=1):
            title = str(item.get("title") or f"Task {idx}").strip()
            intent = str(item.get("intent") or "Focus on the key issues of the topic").strip()
            query = str(item.get("query") or state.research_topic).strip()

            if not query:
                query = state.research_topic or ""

            task = TodoItem(
                id=idx,
                title=title,
                intent=intent,
                query=query,
            )
            todo_items.append(task)

        state.todo_items = todo_items

        titles = [task.title for task in todo_items]
        logger.info("Planner produced %d tasks: %s", len(todo_items), titles)
        return todo_items

    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        """Create a minimal fallback task when planning failed."""

        return TodoItem(
            id=1,
            title="Background Research",
            intent="Collect the core background and latest developments of the topic",
            query=f"{state.research_topic} latest developments" if state.research_topic else "Background Research",
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _extract_tasks(self, raw_response: str) -> list[dict[str, Any]]:
        """Parse planner output into a list of task dictionaries."""

        text = raw_response.strip()
        if self._strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        json_payload = self._extract_json_payload(text)
        tasks: list[dict[str, Any]] = []

        if isinstance(json_payload, dict):
            candidate = json_payload.get("tasks")
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        tasks.append(item)
        elif isinstance(json_payload, list):
            for item in json_payload:
                if isinstance(item, dict):
                    tasks.append(item)

        return tasks

    def _extract_json_payload(self, text: str) -> dict[str, Any] | list | None:
        """Try to locate and parse a JSON object or array from the text."""

        # Try to find the complete JSON (from the first { to the last matching })
        # Using stack to match braces
        def find_matching_brace(text: str, start_pos: int, open_char: str, close_char: str):
            if open_char not in text[start_pos:]:
                return None
            start = text.find(open_char, start_pos)
            depth = 0
            for i in range(start, len(text)):
                if text[i] == open_char:
                    depth += 1
                elif text[i] == close_char:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
            return None

        # Try to find the object
        json_str = find_matching_brace(text, 0, "{", "}")
        if json_str:
            try:
                result: dict[str, Any] | list[Any] = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                pass

        # Try to find the array
        json_str = find_matching_brace(text, 0, "[", "]")
        if json_str:
            try:
                result_arr: dict[str, Any] | list[Any] = json.loads(json_str)
                return result_arr
            except json.JSONDecodeError:
                pass

        return None
