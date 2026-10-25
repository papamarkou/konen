from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from kimi_sdk import GenerateResult, Message, TokenUsage, ToolCall

import konen.backend.kimi_engine as kimi_engine_module
from konen import KimiEngine


class _FakeKimi:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.thinking_effort: str | None = None

    def with_thinking(self, effort: str) -> "_FakeKimi":
        self.thinking_effort = effort
        return self


@contextmanager
def _engine(
    monkeypatch: pytest.MonkeyPatch,
    result: GenerateResult,
    config: dict[str, Any] | None = None,
) -> Iterator[tuple[KimiEngine, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []
    client = object()

    async def fake_generate(**kwargs: Any) -> GenerateResult:
        calls.append(kwargs)
        return result

    monkeypatch.setattr(KimiEngine, "_initialize_client", lambda self: client)
    monkeypatch.setattr(kimi_engine_module, "generate", fake_generate)

    engine = KimiEngine(
        model_family="kimi",
        model_name="kimi-k2.6",
        config={} if config is None else config,
        uid="kimi-analysis",
    )
    try:
        yield engine, calls
    finally:
        engine._runner.close()


def _result(
    *,
    text: str = "The posterior probability remains above 0.75.",
    result_id: str | None = "chatcmpl_01HXYZ123",
    trace_id: str | None = "trace_01HXYZ123",
    usage: TokenUsage | None = None,
) -> GenerateResult:
    return GenerateResult(
        id=result_id,
        message=Message(role="assistant", content=text),
        usage=usage,
        trace_id=trace_id,
    )


def test_kimi_engine_requires_kimi_family() -> None:
    with pytest.raises(ValueError, match="requires model_family='kimi'"):
        KimiEngine(model_family="gpt", model_name="kimi-k2.6")


def test_kimi_engine_client_default(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[_FakeKimi] = []

    def fake_kimi(**kwargs: Any) -> _FakeKimi:
        client = _FakeKimi(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(kimi_engine_module, "Kimi", fake_kimi)

    engine = KimiEngine(
        model_family="kimi", model_name="kimi-k2.6", uid="kimi-analysis"
    )
    try:
        assert engine.client is clients[0]
        assert clients[0].kwargs == {"model": "kimi-k2.6"}
    finally:
        engine._runner.close()


def test_kimi_engine_client_config(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[_FakeKimi] = []

    def fake_kimi(**kwargs: Any) -> _FakeKimi:
        client = _FakeKimi(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(kimi_engine_module, "Kimi", fake_kimi)

    engine = KimiEngine(
        model_family="kimi",
        model_name="kimi-k2.6",
        config={
            "api_key": "test-api-key",
            "base_url": "https://api.moonshot.ai/v1",
            "thinking_effort": "high",
        },
        uid="kimi-analysis",
    )
    try:
        assert engine.client is clients[0]
        assert clients[0].kwargs == {
            "model": "kimi-k2.6",
            "api_key": "test-api-key",
            "base_url": "https://api.moonshot.ai/v1",
        }
        assert clients[0].thinking_effort == "high"
    finally:
        engine._runner.close()


@pytest.mark.parametrize("thinking_effort", ["minimal", "very_high", 1])
def test_kimi_engine_invalid_thinking_effort(
    monkeypatch: pytest.MonkeyPatch, thinking_effort: Any
) -> None:
    monkeypatch.setattr(
        kimi_engine_module, "Kimi", lambda **kwargs: _FakeKimi(**kwargs)
    )

    with pytest.raises(ValueError, match="thinking_effort.*must be one of"):
        KimiEngine(
            model_family="kimi",
            model_name="kimi-k2.6",
            config={"thinking_effort": thinking_effort},
            uid="kimi-analysis",
        )


def test_kimi_engine_prepare_provider_state() -> None:
    engine = object.__new__(KimiEngine)
    states = [
        {
            "user_message": {
                "role": "user",
                "content": "Assess robustness under the baseline prior.",
            },
            "assistant_message": {
                "role": "assistant",
                "content": "The baseline posterior probability is 0.82.",
            },
        },
        {
            "user_message": {"role": "user", "content": "Now widen the prior scale."},
            "assistant_message": {
                "role": "assistant",
                "content": "The posterior probability decreases to 0.79.",
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


def test_kimi_engine_prepare_provider_state_empty() -> None:
    engine = object.__new__(KimiEngine)

    assert engine.prepare_provider_state([]) is None


@pytest.mark.parametrize(
    "state, message",
    [
        (
            {"assistant_message": {"role": "assistant", "content": "Answer"}},
            "'user_message' dictionary",
        ),
        (
            {"user_message": {"role": "user", "content": "Prompt"}},
            "'assistant_message' dictionary",
        ),
    ],
)
def test_kimi_engine_invalid_turn_state(state: dict[str, Any], message: str) -> None:
    engine = object.__new__(KimiEngine)

    with pytest.raises(ValueError, match=message):
        engine.prepare_provider_state([state])


def test_kimi_engine_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = TokenUsage(input_other=100, output=50, input_cache_read=30)

    with _engine(monkeypatch, _result(usage=usage)) as (engine, calls):
        result = engine.generate_response(
            prompt=(
                "Assess whether the posterior probability remains above 0.75 "
                "under the wider prior."
            ),
            system_instruction=(
                "Evaluate uncertainty using Bayesian inference and report assumptions."
            ),
        )

        assert len(calls) == 1
        assert calls[0]["chat_provider"] is engine.client
        assert calls[0]["system_prompt"] == (
            "Evaluate uncertainty using Bayesian inference and report assumptions."
        )
        assert calls[0]["tools"] == []
        assert [
            message.model_dump(mode="json", exclude_none=True)
            for message in calls[0]["history"]
        ] == [
            {
                "role": "user",
                "content": (
                    "Assess whether the posterior probability remains above 0.75 "
                    "under the wider prior."
                ),
            }
        ]
        assert result.text == "The posterior probability remains above 0.75."
        assert result.metadata == {
            "engine_uid": "kimi-analysis",
            "model_family": "kimi",
            "model_name": "kimi-k2.6",
            "provider_response_id": "chatcmpl_01HXYZ123",
            "provider_trace_id": "trace_01HXYZ123",
            "token_usage": {
                "input_token_count": 130,
                "output_token_count": 50,
                "total_token_count": 180,
                "cached_token_count": 30,
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
                "content": "The posterior probability remains above 0.75.",
            },
        }


def test_kimi_engine_generate_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_state = {
        "messages": [
            {"role": "user", "content": "Assess robustness under the baseline prior."},
            {
                "role": "assistant",
                "content": "The baseline posterior probability is 0.82.",
            },
        ]
    }

    with _engine(monkeypatch, _result()) as (engine, calls):
        engine.generate_response(
            prompt="Reassess the conclusion after widening the prior scale.",
            provider_state=provider_state,
        )

        assert [
            message.model_dump(mode="json", exclude_none=True)
            for message in calls[0]["history"]
        ] == [
            provider_state["messages"][0],
            provider_state["messages"][1],
            {
                "role": "user",
                "content": "Reassess the conclusion after widening the prior scale.",
            },
        ]
        assert calls[0]["system_prompt"] == ""

        calls[0]["history"][0].content[0].text = "Changed"
        assert provider_state["messages"][0]["content"] == (
            "Assess robustness under the baseline prior."
        )


def test_kimi_engine_usage_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    with _engine(monkeypatch, _result(usage=None)) as (engine, _):
        result = engine.generate_response(
            prompt="Assess robustness under the wider prior."
        )

        assert "token_usage" not in result.metadata


def test_kimi_engine_optional_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    with _engine(monkeypatch, _result(result_id=None, trace_id=None)) as (engine, _):
        result = engine.generate_response(
            prompt="Assess robustness under the wider prior."
        )

        assert "provider_response_id" not in result.metadata
        assert "provider_trace_id" not in result.metadata


def test_kimi_engine_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    with _engine(monkeypatch, _result(text="")) as (engine, _):
        result = engine.generate_response(
            prompt="Assess robustness under the wider prior."
        )

        assert result.text is None


def test_kimi_engine_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    message = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id="call_01HXYZ123",
                function=ToolCall.FunctionBody(
                    name="search_literature", arguments="{}"
                ),
            )
        ],
    )
    result = GenerateResult(
        id="chatcmpl_01HXYZ123", message=message, usage=None, trace_id="trace_01HXYZ123"
    )

    with _engine(monkeypatch, result) as (engine, _):
        with pytest.raises(RuntimeError, match="returned tool calls"):
            engine.generate_response(
                prompt="Search the literature for robustness analyses."
            )


@pytest.mark.parametrize(
    "provider_state",
    [
        {},
        {"messages": []},
        {"messages": ["not-a-message"]},
        {"messages": [{"role": "unknown", "content": "Context"}]},
    ],
)
def test_kimi_engine_invalid_provider_state(
    monkeypatch: pytest.MonkeyPatch, provider_state: dict[str, Any]
) -> None:
    with _engine(monkeypatch, _result()) as (engine, _):
        with pytest.raises((ValueError, TypeError)):
            engine.generate_response(
                prompt="Assess robustness under the wider prior.",
                provider_state=provider_state,
            )


def test_kimi_engine_repeated_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    client = object()

    async def fake_generate(**kwargs: Any) -> GenerateResult:
        calls.append(kwargs)
        turn = len(calls)
        return GenerateResult(
            id=f"chatcmpl_0{turn}",
            message=Message(
                role="assistant", content=f"Posterior assessment for turn {turn}."
            ),
            usage=None,
            trace_id=f"trace_0{turn}",
        )

    monkeypatch.setattr(KimiEngine, "_initialize_client", lambda self: client)
    monkeypatch.setattr(kimi_engine_module, "generate", fake_generate)

    engine = KimiEngine(
        model_family="kimi", model_name="kimi-k2.6", uid="kimi-analysis"
    )
    try:
        first = engine.generate_response("Assess the baseline posterior.")
        second = engine.generate_response("Assess the wider-prior posterior.")

        assert first.text == "Posterior assessment for turn 1."
        assert second.text == "Posterior assessment for turn 2."
        assert len(calls) == 2
    finally:
        engine._runner.close()
