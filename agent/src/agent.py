"""Base agent class"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from .llm import LLM
from .message import Message


class Agent(ABC):
    """Base agent class"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: str | None = None,
        **kwargs,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """Run the agent synchronously and return the full response."""
        pass

    @abstractmethod
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """Run the agent and yield response chunks as they arrive."""
        pass

    def add_message(self, message: Message):
        """Add message to history"""
        self._history.append(message)

    def clear_history(self):
        """Clear history"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """Get history"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, system_prompt={self.system_prompt}, llm={self.llm.model_id})"

    def __repr__(self) -> str:
        return self.__str__()
