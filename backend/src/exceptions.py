"""Backend-layer exception hierarchy."""


class DeepResearchError(Exception):
    """Base exception for all deep-research backend errors."""


class PlanningError(DeepResearchError):
    """Raised when task planning fails."""


class SearchError(DeepResearchError):
    """Raised when web search dispatch fails."""


class ReviewError(DeepResearchError):
    """Raised when the reviewer service fails."""


class SummarizationError(DeepResearchError):
    """Raised when task summarization fails."""


class ReportError(DeepResearchError):
    """Raised when final report generation fails."""
