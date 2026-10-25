import copy
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic

from .llm_engine import LLMEngine
from .llm_response import LLMResponse


@dataclass(slots=True)
class ClaudeEngine(LLMEngine):
    """Anthropic Claude execution engine."""

    def __post_init__(self) -> None:
        if self.model_family != "claude":
            raise ValueError("ClaudeEngine requires model_family='claude'.")
        super().__post_init__()

    def _initialize_client(self) -> Anthropic:
        api_key = self.config.get("api_key")
        if api_key is None:
            return Anthropic()
        return Anthropic(api_key=api_key)

    def prepare_provider_state(
        self, states: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Prepare Anthropic message history from prior provider states."""
        if not states:
            return None

        messages: list[dict[str, Any]] = []
        for state in states:
            user_message = state.get("user_message")
            assistant_message = state.get("assistant_message")

            if not isinstance(user_message, dict):
                raise ValueError(
                    "Claude provider state must contain a 'user_message' dictionary."
                )
            if not isinstance(assistant_message, dict):
                raise ValueError(
                    "Claude provider state must contain an "
                    "'assistant_message' dictionary."
                )

            messages.append(copy.deepcopy(user_message))
            messages.append(copy.deepcopy(assistant_message))

        return {"messages": messages}

    def generate_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        provider_state: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response with the Anthropic Messages API."""
        messages = self._previous_messages(provider_state)
        user_message = {"role": "user", "content": prompt}
        messages.append(copy.deepcopy(user_message))

        message_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self._max_tokens(),
            "messages": messages,
        }

        if system_instruction is not None:
            message_kwargs["system"] = system_instruction

        thinking = self.config.get("thinking")
        if thinking is not None:
            if not isinstance(thinking, dict):
                raise ValueError(
                    "ClaudeEngine.config['thinking'] must be a dictionary."
                )
            message_kwargs["thinking"] = copy.deepcopy(thinking)

        message = self.client.messages.create(**message_kwargs)

        stop_reason = message.stop_reason
        if stop_reason is None:
            raise RuntimeError("Claude message did not return a stop reason.")

        metadata = self._build_response_metadata()
        metadata["provider_response_id"] = message.id
        metadata["stop_reason"] = stop_reason

        if message.stop_sequence is not None:
            metadata["stop_sequence"] = message.stop_sequence

        usage = message.usage
        cache_creation_tokens = usage.cache_creation_input_tokens or 0
        cached_tokens = usage.cache_read_input_tokens or 0
        input_tokens = usage.input_tokens + cache_creation_tokens + cached_tokens

        token_usage: dict[str, int] = {
            "input_token_count": input_tokens,
            "output_token_count": usage.output_tokens,
            "total_token_count": input_tokens + usage.output_tokens,
        }

        if usage.cache_creation_input_tokens is not None:
            token_usage["cache_creation_token_count"] = (
                usage.cache_creation_input_tokens
            )

        if usage.cache_read_input_tokens is not None:
            token_usage["cached_token_count"] = usage.cache_read_input_tokens

        output_details = getattr(usage, "output_tokens_details", None)
        if output_details is not None:
            thinking_tokens = getattr(output_details, "thinking_tokens", None)
            if thinking_tokens is not None:
                token_usage["thinking_token_count"] = thinking_tokens

        metadata["token_usage"] = token_usage

        if stop_reason in {"tool_use", "pause_turn"}:
            raise RuntimeError(
                "Claude message requires continuation handling that Konen "
                f"does not yet implement; stop_reason is '{stop_reason}'."
            )

        supported_stop_reasons = {
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "refusal",
            "model_context_window_exceeded",
        }
        if stop_reason not in supported_stop_reasons:
            raise RuntimeError(f"Unexpected Claude stop reason '{stop_reason}'.")

        text_parts = [block.text for block in message.content if block.type == "text"]
        text = "".join(text_parts) or None

        assistant_content = [
            block.model_dump(mode="json", exclude_none=True)
            for block in message.content
        ]
        assistant_message = {"role": "assistant", "content": assistant_content}

        return LLMResponse(
            text=text,
            metadata=metadata,
            provider_state={
                "user_message": user_message,
                "assistant_message": assistant_message,
            },
        )

    def _max_tokens(self) -> int:
        """Return the validated maximum output token count."""
        max_tokens = self.config.get("max_tokens")

        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
        ):
            raise ValueError(
                "ClaudeEngine.config['max_tokens'] must be a positive integer."
            )

        return max_tokens

    @staticmethod
    def _previous_messages(
        provider_state: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Validate and return prior Anthropic messages."""
        if provider_state is None:
            return []

        messages = provider_state.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "Claude prepared provider state must contain a non-empty "
                "'messages' list."
            )

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Claude provider state messages must be dictionaries.")

            if message.get("role") not in {"user", "assistant"}:
                raise ValueError(
                    "Claude provider state messages must have role 'user' "
                    "or 'assistant'."
                )

            if "content" not in message:
                raise ValueError(
                    "Claude provider state messages must contain 'content'."
                )

        return copy.deepcopy(messages)
