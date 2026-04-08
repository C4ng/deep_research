from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional
import os

class SearchAPI(Enum):
    TAVILY = "tavily"

   
class Configuration(BaseModel):
    """Configuration options for the deep research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Max search-review iterations per task (reflection loop cap)",
    )

    max_tasks_per_topic: int = Field(
        default=3,
        title="Max Tasks Per Topic",
        description="Max number of sub-tasks the planner can execute per research topic",
    )

    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        title="Search API",
        description="Web search API to use",
    )
  
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
  
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )

    use_tool_calling: bool = Field(
        default=False,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )

    llm_api_key: str = Field(
        default=None,
        title="LLM API Key",
        description="API key when using custom OpenAI-compatible services",
    )

    llm_base_url: str = Field(
        default=None,
        title="LLM Base URL",
        description="Base URL when using custom OpenAI-compatible services",
    )

    llm_model_id: str = Field(
        default=None,
        title="LLM Model ID",
        description="Model identifier for custom OpenAI-compatible services",
    )

    cors_allowed_origins: str = Field(
        default="http://localhost:3000",
        title="CORS Allowed Origins",
        description="Comma-separated list of allowed CORS origins",
    )

    @classmethod
    def from_env(cls, overrides: Optional[dict[str, Any]] = None) -> "Configuration":
        """Create a configuration object using environment variables and overrides."""

        raw_values: dict[str, Any] = {}

        # Load values from environment variables based on field names
        for field_name in cls.model_fields.keys():
            env_key = field_name.upper()
            if env_key in os.environ:
                raw_values[field_name] = os.environ[env_key]

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    raw_values[key] = value

        return cls(**raw_values)
