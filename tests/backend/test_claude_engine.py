from types import SimpleNamespace
from typing import Any

import pytest

import konen.backend.claude_engine as claude_engine_module
from konen import ClaudeEngine


class _FakeBlock:
    def __init__(self, block_type: str, **data: Any) -> None:
        self.type = block_type
        for name, value in data.items():
            setattr(self, name, value)

    def model_dump(
        self, mode: str = "json", exclude_none: bool = True
    ) -> dict[str, Any]:
        return {
            name: value
            for name, value in vars(self).items()
            if not exclude_none or value is not None
        }


class _FakeMessages:
    def __init__(self, message: SimpleNamespace) -> None:
        self.message = message
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.message


class _FakeClient:
    def __init__(self, message: SimpleNamespace) -> None:
        self.messages = _FakeMessages(message)


def _usage(
    *,
    cache_creation_input_tokens: int | None = 20,
    cache_read_input_tokens: int | None = 30,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        output_tokens_details=SimpleNamespace(thinking_tokens=18),
    )


def _message(
    *,
    stop_reason: str | None = "end_turn",
    stop_sequence: str | None = None,
    message_id: str = "msg_01HXYZ123",
    content: list[_FakeBlock] | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    if content is None:
        content = [
            _FakeBlock("text", text="The posterior probability remains above 0.75.")
        ]
    return SimpleNamespace(
        id=message_id,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence,
        content=content,
        usage=_usage() if usage is None else usage,
    )


def _engine(
    monkeypatch: pytest.MonkeyPatch,
    message: SimpleNamespace,
    config: dict[str, Any] | None = None,
) -> tuple[ClaudeEngine, _FakeClient]:
    client = _FakeClient(message)
    monkeypatch.setattr(ClaudeEngine, "_initialize_client", lambda self: client)
    engine = ClaudeEngine(
        model_family="claude",
        model_name="claude-sonnet-5",
        config={"max_tokens": 16000} if config is None else config,
        uid="claude-analysis",
    )
    return engine, client


def test_claude_engine_requires_claude_family() -> None:
    with pytest.raises(ValueError, match="requires model_family='claude'"):
        ClaudeEngine(
            model_family="gpt",
            model_name="claude-sonnet-5",
            config={"max_tokens": 16000},
        )


def test_claude_engine_client_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    client = object()

    def fake_anthropic(**kwargs: Any) -> object:
        calls.append(kwargs)
        return client

    monkeypatch.setattr(claude_engine_module, "Anthropic", fake_anthropic)

    engine = ClaudeEngine(
        model_family="claude",
        model_name="claude-sonnet-5",
        config={"max_tokens": 16000},
        uid="claude-analysis",
    )

    assert engine.client is client
    assert calls == [{}]


def test_claude_engine_client_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    client = object()

    def fake_anthropic(**kwargs: Any) -> object:
        calls.append(kwargs)
        return client

    monkeypatch.setattr(claude_engine_module, "Anthropic", fake_anthropic)

    engine = ClaudeEngine(
        model_family="claude",
        model_name="claude-sonnet-5",
        config={"max_tokens": 16000, "api_key": "test-api-key"},
        uid="claude-analysis",
    )

    assert engine.client is client
    assert calls == [{"api_key": "test-api-key"}]


def test_claude_engine_prepare_provider_state() -> None:
    engine = object.__new__(ClaudeEngine)
    states = [
        {
            "user_message": {
                "role": "user",
                "content": "Assess robustness under the baseline prior.",
            },
            "assistant_message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The baseline posterior probability is 0.82.",
                    }
                ],
            },
        },
        {
            "user_message": {"role": "user", "content": "Now widen the prior scale."},
            "assistant_message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The posterior probability decreases to 0.79.",
                    }
                ],
            },
        },
    ]

    prepared = engine.prepare_provider_state(states)

    assert prepared == {
        "messages": [
            states[0]["user_message"],
            states[0]["assistant_message"],
            states[1]["user_message"],
            states[1]["assistant_message"],
        ]
    }
    assert prepared is not None
    prepared["messages"][0]["content"] = "Changed"
    assert states[0]["user_message"]["content"] == (
        "Assess robustness under the baseline prior."
    )


def test_claude_engine_prepare_provider_state_empty() -> None:
    engine = object.__new__(ClaudeEngine)

    assert engine.prepare_provider_state([]) is None


@pytest.mark.parametrize(
    "state, message",
    [
        (
            {"assistant_message": {"role": "assistant", "content": []}},
            "'user_message' dictionary",
        ),
        (
            {"user_message": {"role": "user", "content": "Prompt"}},
            "'assistant_message' dictionary",
        ),
    ],
)
def test_claude_engine_invalid_turn_state(state: dict[str, Any], message: str) -> None:
    engine = object.__new__(ClaudeEngine)

    with pytest.raises(ValueError, match=message):
        engine.prepare_provider_state([state])


def test_claude_engine_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    content = [
        _FakeBlock(
            "thinking",
            thinking="I should compare the posterior under both prior scales.",
            signature="thinking-signature",
        ),
        _FakeBlock("text", text="The posterior probability remains above 0.75."),
        _FakeBlock("text", text=" The conclusion is robust to the wider prior."),
    ]
    engine, client = _engine(
        monkeypatch,
        _message(content=content),
        config={"max_tokens": 16000, "thinking": {"type": "adaptive"}},
    )

    result = engine.generate_response(
        prompt=(
            "Assess whether the posterior probability remains above 0.75 "
            "under the wider prior."
        ),
        system_instruction=(
            "Evaluate uncertainty using Bayesian inference and report assumptions."
        ),
    )

    assert client.messages.calls == [
        {
            "model": "claude-sonnet-5",
            "max_tokens": 16000,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Assess whether the posterior probability remains above 0.75 "
                        "under the wider prior."
                    ),
                }
            ],
            "system": (
                "Evaluate uncertainty using Bayesian inference and report assumptions."
            ),
            "thinking": {"type": "adaptive"},
        }
    ]
    assert result.text == (
        "The posterior probability remains above 0.75. "
        "The conclusion is robust to the wider prior."
    )
    assert result.metadata == {
        "engine_uid": "claude-analysis",
        "model_family": "claude",
        "model_name": "claude-sonnet-5",
        "provider_response_id": "msg_01HXYZ123",
        "stop_reason": "end_turn",
        "token_usage": {
            "input_token_count": 150,
            "output_token_count": 50,
            "total_token_count": 200,
            "cache_creation_token_count": 20,
            "cached_token_count": 30,
            "thinking_token_count": 18,
        },
    }
    assert result.provider_state == {
        "user_message": {
            "role": "user",
            "content": (
                "Assess whether the posterior probability remains above 0.75 "
                "under the wider prior."
            ),
        },
        "assistant_message": {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": (
                        "I should compare the posterior under both prior scales."
                    ),
                    "signature": "thinking-signature",
                },
                {
                    "type": "text",
                    "text": "The posterior probability remains above 0.75.",
                },
                {
                    "type": "text",
                    "text": " The conclusion is robust to the wider prior.",
                },
            ],
        },
    }


def test_claude_engine_generate_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, client = _engine(monkeypatch, _message())
    provider_state = {
        "messages": [
            {"role": "user", "content": "Assess robustness under the baseline prior."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The baseline posterior probability is 0.82.",
                    }
                ],
            },
        ]
    }

    engine.generate_response(
        prompt="Reassess the conclusion after widening the prior scale.",
        provider_state=provider_state,
    )

    assert client.messages.calls == [
        {
            "model": "claude-sonnet-5",
            "max_tokens": 16000,
            "messages": [
                provider_state["messages"][0],
                provider_state["messages"][1],
                {
                    "role": "user",
                    "content": "Reassess the conclusion after widening the prior scale.",
                },
            ],
        }
    ]
    client.messages.calls[0]["messages"][0]["content"] = "Changed"
    assert provider_state["messages"][0]["content"] == (
        "Assess robustness under the baseline prior."
    )


def test_claude_engine_stop_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(
        monkeypatch, _message(stop_reason="stop_sequence", stop_sequence="END")
    )

    result = engine.generate_response(
        prompt="Summarize the robustness assessment and finish with END."
    )

    assert result.metadata["stop_reason"] == "stop_sequence"
    assert result.metadata["stop_sequence"] == "END"


def test_claude_engine_usage_without_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(
        monkeypatch,
        _message(
            usage=_usage(cache_creation_input_tokens=None, cache_read_input_tokens=None)
        ),
    )

    result = engine.generate_response(prompt="Assess robustness under the wider prior.")

    assert result.metadata["token_usage"] == {
        "input_token_count": 100,
        "output_token_count": 50,
        "total_token_count": 150,
        "thinking_token_count": 18,
    }


def test_claude_engine_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(
        monkeypatch,
        _message(
            content=[
                _FakeBlock(
                    "thinking",
                    thinking="I should assess posterior sensitivity.",
                    signature="thinking-signature",
                )
            ]
        ),
    )

    result = engine.generate_response(prompt="Assess robustness under the wider prior.")

    assert result.text is None


def test_claude_engine_invalid_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(
        monkeypatch, _message(), config={"max_tokens": 16000, "thinking": "adaptive"}
    )

    with pytest.raises(ValueError, match="thinking.*must be a dictionary"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


@pytest.mark.parametrize("max_tokens", [None, 0, -1, True, 1.5, "16000"])
def test_claude_engine_invalid_max_tokens(
    monkeypatch: pytest.MonkeyPatch, max_tokens: Any
) -> None:
    engine, _ = _engine(monkeypatch, _message(), config={"max_tokens": max_tokens})

    with pytest.raises(ValueError, match="max_tokens.*positive integer"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


@pytest.mark.parametrize(
    "provider_state",
    [
        {},
        {"messages": []},
        {"messages": ["not-a-message"]},
        {"messages": [{"role": "system", "content": "Context"}]},
        {"messages": [{"role": "user"}]},
    ],
)
def test_claude_engine_invalid_provider_state(
    monkeypatch: pytest.MonkeyPatch, provider_state: dict[str, Any]
) -> None:
    engine, _ = _engine(monkeypatch, _message())

    with pytest.raises(ValueError):
        engine.generate_response(
            prompt="Assess robustness under the wider prior.",
            provider_state=provider_state,
        )


def test_claude_engine_missing_stop_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(monkeypatch, _message(stop_reason=None))

    with pytest.raises(RuntimeError, match="did not return a stop reason"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


@pytest.mark.parametrize("stop_reason", ["tool_use", "pause_turn"])
def test_claude_engine_continuation_stop_reason(
    monkeypatch: pytest.MonkeyPatch, stop_reason: str
) -> None:
    engine, _ = _engine(monkeypatch, _message(stop_reason=stop_reason))

    with pytest.raises(RuntimeError, match=f"stop_reason is '{stop_reason}'"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


@pytest.mark.parametrize(
    "stop_reason",
    [
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "refusal",
        "model_context_window_exceeded",
    ],
)
def test_claude_engine_supported_stop_reason(
    monkeypatch: pytest.MonkeyPatch, stop_reason: str
) -> None:
    engine, _ = _engine(monkeypatch, _message(stop_reason=stop_reason))

    result = engine.generate_response(prompt="Assess robustness under the wider prior.")

    assert result.metadata["stop_reason"] == stop_reason


def test_claude_engine_unexpected_stop_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(monkeypatch, _message(stop_reason="unknown"))

    with pytest.raises(RuntimeError, match="Unexpected Claude stop reason"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")
