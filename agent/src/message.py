from datetime import UTC, datetime
from typing import Any, Literal

import tiktoken
from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    role: MessageRole
    content: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    # TODO: Add tool support later
    tool_call_id: str | None = None
    name: str | None = None

    def token_count(self, model: str = "gemini-2.5-flash") -> int:
        """
        Calculates tokens used by this message.
        Note: API providers add a few tokens for message formatting (role, etc.)
        """
        if not self.content:
            return 0

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for newer or unknown models
            encoding = tiktoken.get_encoding("cl100k_base")

        # Basic tokens in the content
        tokens = len(encoding.encode(self.content))

        # SOTA models add overhead for the role and name
        # We add 4 tokens for the message structure per OpenAI's guidelines
        tokens += 4
        if self.name:
            tokens += 1  # Name overhead

        return tokens

    def to_dict(self) -> dict[str, Any]:
        """OpenAI API compatibility"""
        data = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        return data
