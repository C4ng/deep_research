"""Task review / reflection utilities."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.src.agents.simple_agent import SimpleAgent
from backend.src.cache import llm_cache
from backend.src.models import SummaryState, TaskReview, TodoItem
from backend.src.utils import strip_thinking_tokens

logger = logging.getLogger(__name__)


class ReviewerService:
    """Uses a dedicated agent to reflect on gathered context before summarization."""

    def __init__(self, reviewer_agent: SimpleAgent, strip_thinking_tokens: bool) -> None:
        self._agent = reviewer_agent
        self._strip_thinking_tokens = strip_thinking_tokens
        self._fallback_marker = "[FALLBACK_REVIEW]"

    def review_task(self, state: SummaryState, task: TodoItem, context: str) -> TaskReview:
        """Generate a structured rubric-style review for a single task."""

        prompt = (
            f"Research topic: {state.research_topic}\n"
            f"Task name: {task.title}\n"
            f"Task intent: {task.intent}\n"
            f"Search query: {task.query}\n"
            f"Task findings summary:\n{context}\n"
        )

        logger.debug(
            "Reviewer prompt for task %d (truncated): %s",
            task.id,
            prompt[:1000],
        )

        cached = llm_cache.get("reviewer", prompt)
        if cached is not None:
            raw = cached if isinstance(cached, str) else json.dumps(cached)
            logger.info("Reviewer cache hit for task %d", task.id)
        else:
            try:
                response = self._agent.run(prompt)
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.error("Reviewer failed for task %d: %s", task.id, str(exc), exc_info=True)
                return self._fallback_review(f"Reviewer unavailable due to error: {str(exc)}")
            finally:
                self._agent.clear_history()

            raw = (response or "").strip()
        if self._strip_thinking_tokens:
            raw = strip_thinking_tokens(raw)

        logger.info(
            "Reviewer raw output for task %d (truncated): %s",
            task.id,
            raw[:1000],
        )

        review = self._parse_review_from_raw(raw, task.id)
        if review is not None:
            llm_cache.set("reviewer", prompt, json.dumps(review.model_dump(), ensure_ascii=False))
            return review

        # Last resort: strict retry when primary robust parse still fails.
        logger.error(
            "Primary reviewer parse failed for task %d; attempting strict retry. raw=%r",
            task.id,
            raw,
        )
        try:  # pragma: no cover - defensive guardrail
            retry_raw = self._retry_with_strict_json(prompt, task.id)
            if retry_raw:
                retried = self._parse_review_from_raw(retry_raw, task.id, source="strict_retry")
                if retried is not None:
                    llm_cache.set("reviewer", prompt, json.dumps(retried.model_dump(), ensure_ascii=False))
                    return retried
        except Exception as exc:
            logger.error("Reviewer retry failed for task %d: %s", task.id, str(exc), exc_info=True)

        return self._fallback_review(raw or "No reviewer notes available.")

    def _fallback_review(self, notes: str) -> TaskReview:
        """Return a safe default review when parsing or generation fails."""

        return TaskReview(
            coverage_score=1.0,
            reliability_score=1.0,
            clarity_score=1.0,
            overall_score=1.0,
            should_reresearch=False,
            recommendations=[],
            notes=f"{self._fallback_marker} {notes}",
        )

    def _extract_json_payload(self, text: str) -> dict[str, Any]:
        """Extract a JSON object from a possibly wrapped LLM output."""

        text = self._strip_code_fences(text.strip())
        # Fast path: direct JSON
        try:
            result: dict[str, Any] = json.loads(text)
            return result
        except Exception:
            pass

        # Fallback: find outermost braces
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in reviewer output")

        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    parsed: dict[str, Any] = json.loads(candidate)
                    return parsed

        raise ValueError("Unbalanced braces in reviewer output")

    def _strip_code_fences(self, text: str) -> str:
        """Remove optional markdown code fences around JSON responses."""

        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return stripped

    def _retry_with_strict_json(self, original_prompt: str, task_id: int) -> str:
        """
        Retry once with stricter formatting instructions when first parse fails.
        Returns cleaned raw text or empty string on failure.
        """

        strict_prompt = (
            f"{original_prompt}\n\n"
            "IMPORTANT: Return ONE-LINE JSON ONLY. "
            "No markdown fences, no commentary, no extra tokens."
        )
        logger.warning("Retrying reviewer with strict JSON format for task %d", task_id)
        try:
            response = self._agent.run(strict_prompt)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.error("Reviewer strict retry failed for task %d: %s", task_id, str(exc), exc_info=True)
            return ""
        finally:
            self._agent.clear_history()

        raw = self._strip_code_fences((response or "").strip())
        if self._strip_thinking_tokens:
            raw = strip_thinking_tokens(raw)
        return raw

    def is_fallback_review(self, review: TaskReview) -> bool:
        """Detect synthetic fallback reviews to avoid persisting bad stage cache."""
        return isinstance(review.notes, str) and review.notes.startswith(self._fallback_marker)

    def _parse_partial_review(self, raw: str) -> TaskReview | None:
        """
        Best-effort parser for truncated JSON-like reviewer outputs.
        Extracts fields via regex when structured parsing fails.
        """
        text = self._strip_code_fences(raw)
        lowered = text.lower()
        if "coverage_score" not in lowered and "should_reresearch" not in lowered:
            return None

        def _float_field(name: str, default: float) -> float:
            m = re.search(rf'"{name}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
            if not m:
                return default
            value = float(m.group(1))
            return max(0.0, min(1.0, value))

        def _bool_field(name: str, default: bool) -> bool:
            m = re.search(rf'"{name}"\s*:\s*(true|false)', lowered)
            if not m:
                return default
            return m.group(1) == "true"

        coverage = _float_field("coverage_score", 0.5)
        reliability = _float_field("reliability_score", 0.5)
        clarity = _float_field("clarity_score", 0.5)
        overall = _float_field("overall_score", (coverage + reliability + clarity) / 3)
        should_reresearch = _bool_field("should_reresearch", overall < 0.7)

        return TaskReview(
            coverage_score=coverage,
            reliability_score=reliability,
            clarity_score=clarity,
            overall_score=overall,
            should_reresearch=should_reresearch,
            recommendations=self._extract_recommendations(text),
            notes=self._extract_notes(text) or f"Recovered from partial reviewer output: {text[:500]}",
        )

    def _parse_review_from_raw(self, raw: str, task_id: int, source: str = "primary") -> TaskReview | None:
        """
        Parse review from potentially malformed/truncated model output.
        Prefer strict JSON parse, then partial extraction from broken JSON-like text.
        """
        try:
            payload = self._extract_json_payload(raw)
            if not isinstance(payload, dict):
                raise ValueError("Reviewer JSON payload is not an object")
            review = TaskReview(**payload)
            logger.info("Reviewer %s parse succeeded for task %d", source, task_id)
            return review
        except Exception as exc:
            cleaned = self._strip_code_fences((raw or "").strip())
            diagnostics = self._diagnose_jsonish(cleaned)
            logger.warning(
                "Reviewer %s strict JSON parse failed for task %d: %s; diagnostics=%s",
                source,
                task_id,
                str(exc),
                diagnostics,
            )
            partial = self._parse_partial_review(cleaned)
            if partial is not None:
                logger.info("Reviewer %s partial parse succeeded for task %d", source, task_id)
            return partial

    def _diagnose_jsonish(self, text: str) -> dict[str, int]:
        """Return quick diagnostics to explain malformed/truncated JSON-like output."""
        return {
            "open_braces": text.count("{"),
            "close_braces": text.count("}"),
            "open_brackets": text.count("["),
            "close_brackets": text.count("]"),
            "double_quotes": text.count('"'),
        }

    def _extract_recommendations(self, text: str) -> list[str]:
        """
        Best-effort extraction for recommendations even when the array is incomplete.
        """
        recs_start = text.find('"recommendations"')
        if recs_start == -1:
            return []
        tail = text[recs_start:]
        # Capture quoted strings after "recommendations" key.
        values = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', tail)
        if not values:
            return []
        # First match is often the key itself; drop key-like entries.
        filtered = [
            v
            for v in values
            if v
            not in {
                "recommendations",
                "notes",
                "coverage_score",
                "reliability_score",
                "clarity_score",
                "overall_score",
                "should_reresearch",
            }
        ]
        return [v.strip() for v in filtered[:4] if v.strip()]

    def _extract_notes(self, text: str) -> str:
        """Best-effort extraction of notes field from malformed JSON-like output."""
        m = re.search(r'"notes"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
        if not m:
            return ""
        return m.group(1).strip()
