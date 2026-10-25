from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .llm_engine import LLMEngine
from .llm_response import LLMResponse


@dataclass(slots=True)
class GPTEngine(LLMEngine):
    """OpenAI GPT execution engine."""

    def __post_init__(self) -> None:
        if self.model_family != "gpt":
            raise ValueError("GPTEngine requires model_family='gpt'.")
        super().__post_init__()

    def _initialize_client(self) -> OpenAI:
        api_key = self.config.get("api_key")
        if api_key is None:
            return OpenAI()
        return OpenAI(api_key=api_key)

    def generate_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        provider_state: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response with the OpenAI Responses API."""
        response_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": prompt,
            "store": True,
        }

        if system_instruction is not None:
            response_kwargs["instructions"] = system_instruction

        previous_response_id = self._previous_response_id(provider_state)
        if previous_response_id is not None:
            response_kwargs["previous_response_id"] = previous_response_id

        reasoning_effort = self.config.get("reasoning_effort")
        if reasoning_effort is not None:
            response_kwargs["reasoning"] = {"effort": reasoning_effort}

        response = self.client.responses.create(**response_kwargs)

        status = response.status
        if status is None:
            raise RuntimeError("OpenAI response did not return a status.")

        metadata = self._build_response_metadata()
        metadata["response_status"] = status
        metadata["provider_response_id"] = response.id

        usage = response.usage
        if usage is not None:
            token_usage: dict[str, int] = {
                "input_token_count": usage.input_tokens,
                "output_token_count": usage.output_tokens,
                "total_token_count": usage.total_tokens,
            }

            input_details = usage.input_tokens_details
            if input_details is not None:
                cached_tokens = input_details.cached_tokens
                if cached_tokens is not None:
                    token_usage["cached_token_count"] = cached_tokens

            output_details = usage.output_tokens_details
            if output_details is not None:
                reasoning_tokens = output_details.reasoning_tokens
                if reasoning_tokens is not None:
                    token_usage["reasoning_token_count"] = reasoning_tokens

            metadata["token_usage"] = token_usage

        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"OpenAI response ended with status '{status}'.")

        if status in {"queued", "in_progress"}:
            raise RuntimeError(
                f"OpenAI response did not complete synchronously; status is '{status}'."
            )

        if status not in {"completed", "incomplete"}:
            raise RuntimeError(f"Unexpected OpenAI response status '{status}'.")

        text = response.output_text or None

        next_provider_state: dict[str, Any] = {}

        if status == "completed":
            if not isinstance(response.id, str) or not response.id:
                raise RuntimeError("Completed OpenAI response did not return an ID.")

            next_provider_state["response_id"] = response.id

        return LLMResponse(
            text=text, metadata=metadata, provider_state=next_provider_state
        )

    @staticmethod
    def _previous_response_id(provider_state: dict[str, Any] | None) -> str | None:
        """Validate and return the previous OpenAI response identifier."""
        if provider_state is None:
            return None

        response_id = provider_state.get("response_id")

        if not isinstance(response_id, str) or not response_id:
            raise ValueError(
                "OpenAI provider state must contain a non-empty 'response_id' string."
            )

        return response_id
