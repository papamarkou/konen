import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any

from kimi_sdk import Kimi, Message, generate

from .llm_engine import LLMEngine
from .llm_response import LLMResponse


@dataclass(slots=True)
class KimiEngine(LLMEngine):
    """Moonshot Kimi execution engine."""

    _runner: asyncio.Runner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.model_family != "kimi":
            raise ValueError("KimiEngine requires model_family='kimi'.")
        super().__post_init__()
        self._runner = asyncio.Runner()

    def _initialize_client(self) -> Kimi:
        """Initialize the Kimi client and validate thinking effort."""
        client_kwargs: dict[str, Any] = {"model": self.model_name}

        api_key = self.config.get("api_key")
        if api_key is not None:
            client_kwargs["api_key"] = api_key

        base_url = self.config.get("base_url")
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        client = Kimi(**client_kwargs)

        thinking_effort = self.config.get("thinking_effort")
        if thinking_effort is not None:
            allowed_efforts = {"off", "low", "medium", "high", "xhigh", "max"}
            if thinking_effort not in allowed_efforts:
                raise ValueError(
                    "KimiEngine.config['thinking_effort'] must be one of "
                    f"{sorted(allowed_efforts)}."
                )
            client = client.with_thinking(thinking_effort)

        return client

    def prepare_provider_state(
        self, states: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Prepare Kimi message history from prior provider states."""
        if not states:
            return None

        messages: list[dict[str, Any]] = []
        for state in states:
            user_message = state.get("user_message")
            assistant_message = state.get("assistant_message")

            if not isinstance(user_message, dict):
                raise ValueError(
                    "Kimi provider state must contain a 'user_message' dictionary."
                )
            if not isinstance(assistant_message, dict):
                raise ValueError(
                    "Kimi provider state must contain an "
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
        """Generate a response with the Kimi SDK."""
        messages = self._previous_messages(provider_state)
        user_message = Message(role="user", content=prompt)
        messages.append(user_message)

        result = self._runner.run(
            generate(
                chat_provider=self.client,
                system_prompt=system_instruction or "",
                tools=[],
                history=messages,
            )
        )

        if result.message.tool_calls:
            raise RuntimeError(
                "Kimi returned tool calls, but Konen tool continuation is not "
                "implemented."
            )

        metadata = self._build_response_metadata()

        if result.id is not None:
            metadata["provider_response_id"] = result.id

        if result.trace_id is not None:
            metadata["provider_trace_id"] = result.trace_id

        usage = result.usage
        if usage is not None:
            token_usage: dict[str, int] = {
                "input_token_count": usage.input,
                "output_token_count": usage.output,
                "total_token_count": usage.total,
            }

            if usage.input_cache_read:
                token_usage["cached_token_count"] = usage.input_cache_read

            if usage.input_cache_creation:
                token_usage["cache_creation_token_count"] = usage.input_cache_creation

            metadata["token_usage"] = token_usage

        text = result.message.extract_text() or None

        return LLMResponse(
            text=text,
            metadata=metadata,
            provider_state={
                "user_message": user_message.model_dump(mode="json", exclude_none=True),
                "assistant_message": result.message.model_dump(
                    mode="json", exclude_none=True
                ),
            },
        )

    @staticmethod
    def _previous_messages(provider_state: dict[str, Any] | None) -> list[Message]:
        """Validate and return prior Kimi messages."""
        if provider_state is None:
            return []

        messages = provider_state.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "Kimi prepared provider state must contain a non-empty 'messages' list."
            )

        validated_messages: list[Message] = []

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Kimi provider state messages must be dictionaries.")
            validated_messages.append(Message.model_validate(message))

        return validated_messages
