from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError


class Tool(ABC):
    """
    Base tool class.
    Forces strict validation before execution.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @property
    @abstractmethod
    def args_schema(self) -> type[BaseModel]:
        """
        Use a Pydantic model class as the schema.
        This provides automatic validation and OpenAI schema generation.
        """
        pass

    @abstractmethod
    def _execute(self, **kwargs) -> str:
        """The actual logic of the tool."""
        pass

    def run(self, parameters: dict[str, Any]) -> str:
        """
        The public entry point. Handles validation automatically.
        """
        try:
            # Validate raw dict against Pydantic schema
            validated_args = self.args_schema(**parameters)
            # Run execution with validated data
            return self._execute(**validated_args.model_dump())
        except ValidationError as e:
            return f"Error: Invalid parameters for tool {self.name}. Details: {str(e)}"
        except Exception as e:
            return f"Error: Execution failed in {self.name}. {str(e)}"

    def to_openai_schema(self) -> dict[str, Any]:
        """
        Uses Pydantic's internal json_schema to generate the OpenAI format.
        This is much more robust than manual dictionary building.
        """
        schema = self.args_schema.model_json_schema()

        # Clean up the Pydantic schema to match OpenAI's 'function' expectations
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }
