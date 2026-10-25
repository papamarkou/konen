from dataclasses import dataclass
from typing import Any

from google import genai

from .llm_engine import LLMEngine
from .llm_response import LLMResponse


@dataclass(slots=True)
class GeminiEngine(LLMEngine):
    """Google Gemini execution engine."""

    def __post_init__(self) -> None:
        if self.model_family != "gemini":
            raise ValueError("GeminiEngine requires model_family='gemini'.")
        super().__post_init__()

    def _initialize_client(self) -> genai.Client:
        api_key = self.config.get("api_key")
        if api_key is None:
            return genai.Client()
        return genai.Client(api_key=api_key)

    def generate_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        provider_state: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response with the Gemini Interactions API."""
        generation_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": prompt,
            "store": True,
        }

        if system_instruction is not None:
            generation_kwargs["system_instruction"] = system_instruction

        previous_interaction_id = self._previous_interaction_id(provider_state)
        if previous_interaction_id is not None:
            generation_kwargs["previous_interaction_id"] = previous_interaction_id

        generation_config: dict[str, Any] = {}

        thinking_level = self.config.get("thinking_level")
        if thinking_level is not None:
            generation_config["thinking_level"] = thinking_level

        if generation_config:
            generation_kwargs["generation_config"] = generation_config

        interaction = self.client.interactions.create(**generation_kwargs)

        status = self._status_value(interaction.status)
        if status is None:
            raise RuntimeError("Gemini interaction did not return a status.")

        metadata = self._build_response_metadata()
        metadata["interaction_status"] = status

        interaction_id = interaction.id
        if interaction_id is not None:
            metadata["provider_response_id"] = interaction_id

        usage = interaction.usage
        if usage is not None:
            token_usage: dict[str, int] = {}

            usage_fields = {
                "input_token_count": usage.total_input_tokens,
                "output_token_count": usage.total_output_tokens,
                "total_token_count": usage.total_tokens,
                "cached_token_count": usage.total_cached_tokens,
                "thought_token_count": usage.total_thought_tokens,
                "tool_use_token_count": usage.total_tool_use_tokens,
            }

            for name, value in usage_fields.items():
                if value is not None:
                    token_usage[name] = value

            if token_usage:
                metadata["token_usage"] = token_usage

        if status in {"failed", "cancelled", "budget_exceeded"}:
            raise RuntimeError(f"Gemini interaction ended with status '{status}'.")

        if status in {"queued", "in_progress", "requires_action"}:
            raise RuntimeError(
                "Gemini interaction did not complete synchronously; "
                f"status is '{status}'."
            )

        if status not in {"completed", "incomplete"}:
            raise RuntimeError(f"Unexpected Gemini interaction status '{status}'.")

        try:
            text = interaction.output_text
        except (ValueError, AttributeError):
            text = None

        next_provider_state: dict[str, Any] = {}

        if status == "completed":
            if not isinstance(interaction_id, str) or not interaction_id:
                raise RuntimeError("Completed Gemini interaction did not return an ID.")

            next_provider_state["interaction_id"] = interaction_id

        return LLMResponse(
            text=text, metadata=metadata, provider_state=next_provider_state
        )

    @staticmethod
    def _previous_interaction_id(provider_state: dict[str, Any] | None) -> str | None:
        """Validate and return the previous Gemini interaction identifier."""
        if provider_state is None:
            return None

        interaction_id = provider_state.get("interaction_id")

        if not isinstance(interaction_id, str) or not interaction_id:
            raise ValueError(
                "Gemini provider state must contain a non-empty "
                "'interaction_id' string."
            )

        return interaction_id

    @staticmethod
    def _status_value(status: Any) -> str | None:
        """Return the Gemini interaction status value."""
        if status is None:
            return None

        return getattr(status, "value", str(status))
