from typing import Any, Iterator

from backend.src.config import Configuration
from agent.src.llm import LLM
from agent.src.agents.simple_agent import SimpleAgent
from backend.src.prompts import (
    todo_planner_system_prompt,
    task_summarizer_instructions,
    report_writer_instructions,
    reviewer_instructions,
)
from backend.src.services.planner import PlanningService
from backend.src.services.summarizer import SummarizationService
from backend.src.services.reporter import ReportingService
from backend.src.services.reviewer import ReviewerService
from backend.src.services.search import dispatch_search, prepare_research_context
from backend.src.models import SummaryState, SummaryStateOutput, TodoItem, TaskStatus, TaskReview
from backend.src.cache import stage_file_cache
from threading import Lock, Thread
from typing import Callable
from queue import Queue, Empty
from pathlib import Path

import hashlib
import logging
logger = logging.getLogger(__name__)

class DeepResearchAgent:
    """Coordinator orchestrating TODO-based research workflow."""

    def __init__(self, config: Configuration | None = None) -> None:
        self.config = config or Configuration.from_env()
        self.llm = self._init_llm()

        self._state_lock = Lock()

        # Planner
        self.todo_agent = self._create_simple_agent(
            name="Todo Planner",
            system_prompt=todo_planner_system_prompt.strip(),
        )
        self.planner = PlanningService(self.todo_agent, self.config)

        # Summarizer
        self._summarizer_factory: Callable[[], SimpleAgent] = lambda: self._create_simple_agent(  # noqa: E501
            name="Task Summarizer",
            system_prompt=task_summarizer_instructions.strip(),
        )
        self.summarizer = SummarizationService(self._summarizer_factory, self.config.strip_thinking_tokens)

        # Reviewer (reflection before summarization)
        self.reviewer_agent = self._create_simple_agent(
            name="Task Reviewer",
            system_prompt=reviewer_instructions.strip(),
        )
        self.reviewer = ReviewerService(self.reviewer_agent, self.config.strip_thinking_tokens)

        # Reporter
        self.report_agent = self._create_simple_agent(
          name="Report Writer",
            system_prompt=report_writer_instructions.strip(),
        )
        self.reporting = ReportingService(self.report_agent, self.config.strip_thinking_tokens)
        self._tool_event_sink: Callable[[dict[str, Any]], None] | None = None

    def _set_tool_event_sink(self, sink: Callable[[dict[str, Any]], None] | None) -> None:
        """Set or clear optional callback for tool events (e.g. search progress). No-op if unused."""
        self._tool_event_sink = sink

    def _init_llm(self) -> LLM:
        llm_kwargs: dict[str, Any] = {
            "temperature": 0.0,
        }

        if self.config.llm_model_id:
            llm_kwargs["model_id"] = self.config.llm_model_id
        if self.config.llm_api_key:
            llm_kwargs["api_key"] = self.config.llm_api_key
        if self.config.llm_base_url:
            llm_kwargs["base_url"] = self.config.llm_base_url

        return LLM(**llm_kwargs)
    
    def _create_simple_agent(self, *, name: str, system_prompt: str) -> SimpleAgent:
        """Instantiate a SimpleAgent for planning tasks."""
        return SimpleAgent(
            name=name,
            llm=self.llm,
            system_prompt=system_prompt,
        )

    def _deserialize_cached_tasks(self, payload: dict[str, Any] | None) -> list[TodoItem]:
        if not payload:
            return []
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list):
            return []
        parsed: list[TodoItem] = []
        for item in raw_tasks:
            if not isinstance(item, dict):
                continue
            try:
                parsed.append(TodoItem(**item))
            except Exception:
                continue
        return parsed
    
    # ------------------------------------------------------------------
    # Shared setup helpers
    # ------------------------------------------------------------------

    def _plan_tasks(self, topic: str, state: SummaryState, *, max_tasks: int = 2) -> None:
        """Load or generate planned tasks and cap at *max_tasks*."""
        cached_planner = stage_file_cache.load(topic, "planner")
        state.todo_items = self._deserialize_cached_tasks(cached_planner)
        if state.todo_items:
            logger.info("Planner cache hit for topic %r with %d tasks", topic, len(state.todo_items))
        else:
            state.todo_items = self.planner.plan_todo_list(state)
            logger.info("Planned %d tasks", len(state.todo_items))

        if not state.todo_items:
            state.todo_items = [self.planner.create_fallback_task(state)]

        if len(state.todo_items) > max_tasks:
            state.todo_items = state.todo_items[:max_tasks]
            logger.info("Truncated planned tasks to first %d tasks", max_tasks)

        stage_file_cache.save(topic, "planner", {"tasks": [t.model_dump() for t in state.todo_items]})

    def _generate_final_report(self, topic: str, state: SummaryState) -> str:
        """Load or generate the final report and persist it."""
        cached_report = stage_file_cache.load(topic, "final_report")
        if cached_report and isinstance(cached_report.get("report"), str):
            report = cached_report["report"]
            logger.info("Final report cache hit for topic %r", topic)
        else:
            report = self.reporting.generate_report(state)
            stage_file_cache.save(topic, "final_report", {"report": report})
        state.structured_report = report
        state.running_summary = report
        self._save_user_report(topic, report)
        return report

    def _save_user_report(self, topic: str, report: str) -> None:
        """Write the final report as a readable markdown file under ``output/``."""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        slug = hashlib.sha256((topic or "").encode()).hexdigest()[:12]
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in (topic or "report")[:60]).strip()
        filename = f"{safe_title}_{slug}.md"
        path = output_dir / filename

        usage = self.llm.usage.to_dict()
        meta_lines = [
            f"- **LLM Calls**: {usage['llm_calls']}",
            f"- **Prompt Tokens**: {usage['prompt_tokens']:,}",
            f"- **Completion Tokens**: {usage['completion_tokens']:,}",
            f"- **Total Tokens**: {usage['total_tokens']:,}",
        ]
        meta_section = "\n---\n\n## Pipeline Metadata\n\n" + "\n".join(meta_lines) + "\n"

        path.write_text(f"# {topic}\n\n{report}\n{meta_section}", encoding="utf-8")
        logger.info("Report saved to %s", path)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, topic: str) -> SummaryStateOutput:
        """Execute the research workflow and return the final report."""
        logger.info("Starting research workflow for topic: %s", topic)
        state = SummaryState(research_topic=topic)
        self._plan_tasks(topic, state)

        logger.info("Executing %d tasks...", len(state.todo_items))
        for idx, task in enumerate(state.todo_items, 1):
            logger.info("Executing task %d/%d: %s", idx, len(state.todo_items), task.title)
            try:
                for _ in self._execute_task(state, task, emit_stream=False):
                    pass
            except Exception as e:
                logger.error("Task %d failed: %s", idx, str(e), exc_info=True)
                task.status = TaskStatus.FAILED
                task.summary = f"Task execution failed: {str(e)}"

        report = self._generate_final_report(topic, state)
        usage = self.llm.usage.to_dict()
        logger.info("Workflow completed. Report length: %d chars | usage: %s", len(report), usage)

        return SummaryStateOutput(
            running_summary=report,
            report_markdown=report,
            todo_items=state.todo_items,
            usage=usage,
        )

    def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
        """Execute the research workflow and stream progress events."""
        yield {"type": "status", "message": "Initializing research workflow..."}

        state = SummaryState(research_topic=topic)
        self._plan_tasks(topic, state)

        channel_map: dict[int, dict[str, Any]] = {}
        for index, task in enumerate(state.todo_items, start=1):
            token = f"task_{task.id}"
            task.stream_token = token
            channel_map[task.id] = {"step": index, "token": token}

        yield {
            "type": "todo_list",
            "tasks": [self._serialize_task(t) for t in state.todo_items],
            "step": 0,
        }

        event_queue: Queue[dict[str, Any]] = Queue()

        def enqueue(
            event: dict[str, Any],
            *,
            task: TodoItem | None = None,
            step_override: int | None = None,
        ) -> None:
            payload = dict(event)
            target_task_id = payload.get("task_id")
            if task is not None:
                target_task_id = task.id
                payload["task_id"] = task.id

            channel = channel_map.get(target_task_id) if target_task_id is not None else None
            if channel:
                payload.setdefault("step", channel["step"])
                payload["stream_token"] = channel["token"]
            if step_override is not None:
                payload["step"] = step_override
            event_queue.put(payload)

        threads: list[Thread] = []

        def worker(task: TodoItem, step: int) -> None:
            try:
                enqueue(
                    {
                        "type": "task_status",
                        "task_id": task.id,
                        "status": TaskStatus.RESEARCHING.value,
                        "title": task.title,
                        "intent": task.intent,
                    },
                    task=task,
                )

                for event in self._execute_task(state, task, emit_stream=True, step=step):
                    enqueue(event, task=task)
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Task execution failed", exc_info=exc)
                enqueue(
                    {
                        "type": "task_status",
                        "task_id": task.id,
                        "status": TaskStatus.FAILED.value,
                        "detail": str(exc),
                        "title": task.title,
                        "intent": task.intent,
                    },
                    task=task,
                )
            finally:
                enqueue({"type": "__task_done__", "task_id": task.id})
        
        for task in state.todo_items:
            step = channel_map.get(task.id, {}).get("step", 0)
            thread = Thread(target=worker, args=(task, step), daemon=True)
            threads.append(thread)
            thread.start()
        
        active_workers = len(state.todo_items)
        finished_workers = 0

        try:
            while finished_workers < active_workers:
                event = event_queue.get()
                if event.get("type") == "__task_done__":
                    finished_workers += 1
                    continue
                yield event

            while True:
                try:
                    event = event_queue.get_nowait()
                except Empty:
                    break
                if event.get("type") != "__task_done__":
                    yield event
        finally:
            self._set_tool_event_sink(None)
            for thread in threads:
                thread.join()

        report = self._generate_final_report(topic, state)
        usage = self.llm.usage.to_dict()
        logger.info("Stream workflow completed. usage: %s", usage)
        yield {"type": "final_report", "report": report}
        yield {"type": "done", "usage": usage}

    # ------------------------------------------------------------------
    # _execute_task: thin orchestrator that delegates to stage helpers
    # ------------------------------------------------------------------

    def _execute_task(
        self,
        state: SummaryState,
        task: TodoItem,
        *,
        emit_stream: bool,
        step: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run search + review + summarization for a single task."""
        logger.info("Executing task %d: %s (query: %s)", task.id, task.title, task.query)
        task.status = TaskStatus.RESEARCHING
        topic = state.research_topic or ""

        cached_search = stage_file_cache.load(topic, "search", task.id)
        cached_review = stage_file_cache.load(topic, "review", task.id)
        cached_summary = stage_file_cache.load(topic, "summary", task.id)
        logger.info(
            "Task %d cache probe: search=%s review=%s summary=%s",
            task.id, bool(cached_search), bool(cached_review), bool(cached_summary),
        )

        # --- Stage 1: search (+ inline reflection loop) ----------------
        search_ctx = self._run_search_stage(
            state, task, topic, cached_search, emit_stream, step,
        )
        if search_ctx is None:
            return
        combined_context, combined_sources, backend, notices, review, loop_reviews = search_ctx

        # Emit search results to SSE clients.
        if emit_stream:
            yield from self._emit_search_events(
                task, combined_sources, combined_context, backend, notices, step,
            )

        # --- Stage 2: review -------------------------------------------
        review = self._resolve_review(
            state, task, topic, review, loop_reviews,
            cached_review, combined_context,
        )
        task.review_notes = review.notes

        if emit_stream:
            yield {"type": "review", "task_id": task.id, "review": review.model_dump(), "step": step}

        logger.info(
            "Review for task %d: overall_score=%.3f should_reresearch=%s",
            task.id, review.overall_score, review.should_reresearch,
        )

        # --- Stage 3: summary ------------------------------------------
        yield from self._run_summary_stage(
            state, task, topic, review, combined_context,
            cached_summary, emit_stream, step,
        )

        task.status = TaskStatus.COMPLETED
        logger.info("Task %d completed: %s", task.id, task.title)

        if emit_stream:
            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": "completed",
                "summary": task.summary,
                "sources_summary": task.sources_summary,
                "review_notes": task.review_notes,
                "step": step,
            }

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _run_search_stage(
        self,
        state: SummaryState,
        task: TodoItem,
        topic: str,
        cached_search: dict[str, Any] | None,
        emit_stream: bool,
        step: int | None,
    ) -> tuple[str, str, str, list[str], TaskReview | None, list[dict[str, Any]]] | None:
        """Execute the search + reflection loop (or load from cache).

        Returns ``(combined_context, combined_sources, backend, notices, review, loop_reviews)``
        or ``None`` when the task should be skipped (no results).
        """
        if cached_search and isinstance(cached_search.get("combined_context"), str):
            logger.info("Search stage cache hit for topic=%r task=%d", topic, task.id)
            combined_context = cached_search.get("combined_context", "")
            combined_sources = cached_search.get("combined_sources", "")
            backend = cached_search.get("backend", "")
            notices: list[str] = list(cached_search.get("notices") or [])
            task.sources_summary = combined_sources
            task.notices = notices
            return combined_context, combined_sources, backend, notices, None, []

        logger.info("Search stage cache miss for topic=%r task=%d", topic, task.id)
        return self._search_loop(state, task, topic, emit_stream, step)

    def _search_loop(
        self,
        state: SummaryState,
        task: TodoItem,
        topic: str,
        emit_stream: bool,
        step: int | None,
    ) -> tuple[str, str, str, list[str], TaskReview | None, list[dict[str, Any]]] | None:
        """Multi-iteration search + reflection loop (capped to 2 iterations)."""
        aggregated_contexts: list[str] = []
        aggregated_sources: list[str] = []
        loop_trace: list[dict[str, Any]] = []
        loop_reviews: list[dict[str, Any]] = []
        base_query = task.query
        current_query = base_query
        executed_loops = 0
        combined_context = ""
        combined_sources = ""
        backend = ""
        notices: list[str] = []
        review: TaskReview | None = None

        max_loops = min(self.config.max_web_research_loops, 2)
        for loop_index in range(max_loops):
            executed_loops = loop_index + 1
            logger.info("Search loop %d for task %d (query=%s)", loop_index + 1, task.id, current_query)

            search_result, notices, answer_text, backend = dispatch_search(
                current_query, self.config.search_api, self.config.fetch_full_page, state.research_loop_count,
            )
            task.notices = notices

            if not search_result or not search_result.get("results"):
                task.status = TaskStatus.SKIPPED
                stage_file_cache.save(topic, "search", {"status": TaskStatus.SKIPPED.value, "query": task.query}, task.id)
                return None

            sources_summary, context = prepare_research_context(search_result, answer_text, self.config.fetch_full_page)
            aggregated_contexts.append(context)
            aggregated_sources.append(sources_summary)

            with self._state_lock:
                state.web_research_results.append(context)
                state.sources_gathered.append(sources_summary)
                state.research_loop_count += 1

            combined_context = "\n\n---\n\n".join(aggregated_contexts)
            combined_sources = "\n".join(aggregated_sources)
            task.sources_summary = combined_sources

            reviewer_input = self._build_reviewer_input(state, task, combined_context)
            review = self.reviewer.review_task(state, task, reviewer_input)
            task.review_notes = review.notes
            loop_reviews.append({
                "loop_index": loop_index + 1, "query": current_query,
                "reviewer_input_preview": reviewer_input[:300], "review": review.model_dump(),
            })

            logger.info(
                "Review for task %d: overall_score=%.3f should_reresearch=%s",
                task.id, review.overall_score, review.should_reresearch,
            )
            loop_trace.append({
                "loop_index": loop_index + 1, "query": current_query,
                "overall_score": review.overall_score, "should_reresearch": review.should_reresearch,
                "recommendations": review.recommendations, "notes_preview": (review.notes or "")[:300],
            })

            self._save_search_cache(topic, task, executed_loops, combined_context, combined_sources, backend, notices, loop_trace, status="in_progress")
            stage_file_cache.save(topic, "review", {"review": review.model_dump(), "loop_reviews": loop_reviews}, task.id)

            if review.should_reresearch and (loop_index + 1) < max_loops:
                if review.recommendations:
                    refinement = "; ".join(review.recommendations[:2])
                    current_query = f"{base_query}. Focus also on: {refinement}"
                else:
                    current_query = base_query
                continue
            break

        self._save_search_cache(topic, task, executed_loops, combined_context, combined_sources, backend, notices, loop_trace, status="ok")
        logger.info("Search stage cache saved for topic=%r task=%d (loops=%d)", topic, task.id, executed_loops)
        return combined_context, combined_sources, backend, notices, review, loop_reviews

    def _save_search_cache(
        self, topic: str, task: TodoItem, executed_loops: int,
        combined_context: str, combined_sources: str, backend: str,
        notices: list[str], loop_trace: list[dict[str, Any]], *, status: str,
    ) -> None:
        stage_file_cache.save(topic, "search", {
            "status": status, "query": task.query, "executed_loops": executed_loops,
            "combined_context": combined_context, "combined_sources": combined_sources,
            "backend": backend, "notices": notices, "loop_trace": loop_trace,
        }, task.id)

    @staticmethod
    def _emit_search_events(
        task: TodoItem, combined_sources: str, combined_context: str,
        backend: str, notices: list[str], step: int | None,
    ) -> Iterator[dict[str, Any]]:
        for notice in notices:
            if notice:
                yield {"type": "status", "message": notice, "task_id": task.id, "step": step}
        yield {
            "type": "sources", "task_id": task.id,
            "latest_sources": combined_sources, "raw_context": combined_context,
            "step": step, "backend": backend,
        }

    def _resolve_review(
        self,
        state: SummaryState,
        task: TodoItem,
        topic: str,
        review: TaskReview | None,
        loop_reviews: list[dict[str, Any]],
        cached_review: dict[str, Any] | None,
        combined_context: str,
    ) -> TaskReview:
        """Resolve a TaskReview from cache, existing loop result, or fresh call."""
        if review is not None:
            if not self.reviewer.is_fallback_review(review):
                stage_file_cache.save(topic, "review", {"review": review.model_dump(), "loop_reviews": loop_reviews}, task.id)
            return review

        if cached_review and isinstance(cached_review.get("review"), dict):
            try:
                review = TaskReview(**cached_review["review"])
                legacy_invalid = isinstance(review.notes, str) and review.notes.lstrip().startswith("```")
                if not (self.reviewer.is_fallback_review(review) or legacy_invalid):
                    logger.info("Review stage cache hit for topic=%r task=%d", topic, task.id)
                    return review
                logger.warning("Review stage cache entry is fallback; ignoring for topic=%r task=%d", topic, task.id)
            except Exception:
                pass

        logger.info("Review stage cache miss for topic=%r task=%d", topic, task.id)
        reviewer_input = self._build_reviewer_input(state, task, combined_context)
        review = self.reviewer.review_task(state, task, reviewer_input)
        if not self.reviewer.is_fallback_review(review):
            existing_lr = cached_review.get("loop_reviews") if isinstance(cached_review, dict) else None
            payload: dict[str, Any] = {"review": review.model_dump(), "loop_reviews": existing_lr or []}
            stage_file_cache.save(topic, "review", payload, task.id)
        return review

    def _run_summary_stage(
        self,
        state: SummaryState,
        task: TodoItem,
        topic: str,
        review: TaskReview,
        combined_context: str,
        cached_summary: dict[str, Any] | None,
        emit_stream: bool,
        step: int | None,
    ) -> Iterator[dict[str, Any]]:
        """Produce the task summary (cached or fresh) and yield SSE events."""
        summary_text: str | None = None

        if cached_summary and isinstance(cached_summary.get("summary"), str):
            summary_text = cached_summary["summary"]
            logger.info("Summary stage cache hit for topic=%r task=%d", topic, task.id)
            if emit_stream and summary_text:
                yield {"type": "task_summary_chunk", "task_id": task.id, "content": summary_text, "step": step}
        else:
            logger.info("Summary stage cache miss for topic=%r task=%d", topic, task.id)
            if emit_stream:
                summary_stream, summary_getter = self.summarizer.stream_task_summary(state, task, combined_context)
                try:
                    for chunk in summary_stream:
                        if chunk:
                            yield {"type": "task_summary_chunk", "task_id": task.id, "content": chunk, "step": step}
                finally:
                    summary_text = summary_getter()
            else:
                summary_text = self.summarizer.summarize_task(state, task, combined_context)

            stage_file_cache.save(topic, "summary", {
                "summary": summary_text or "", "review_notes": task.review_notes or "",
                "review_overall_score": review.overall_score,
                "review_should_reresearch": review.should_reresearch,
            }, task.id)

        task.summary = summary_text.strip() if summary_text else "No available information"
    
    def _serialize_task(self, task: TodoItem) -> dict[str, Any]:
        """Convert task dataclass to serializable dict for frontend."""
        return {
            "id": task.id,
            "title": task.title,
            "intent": task.intent,
            "query": task.query,
            "status": task.status,
            "summary": task.summary,
            "sources_summary": task.sources_summary,
            "stream_token": task.stream_token,
        }

    def _build_reviewer_input(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """
        Build concise reviewer input from summarizer output.
        Reviewer should evaluate findings quality, not raw scraped context.
        """
        previous_notes = task.review_notes
        task.review_notes = None
        try:
            draft = self.summarizer.summarize_task(state, task, context)
        finally:
            task.review_notes = previous_notes
        logger.info("Built reviewer input from summarized findings for task %d", task.id)
        return draft


       