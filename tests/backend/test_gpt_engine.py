from types import SimpleNamespace
from typing import Any

import pytest

import konen.backend.gpt_engine as gpt_engine_module
from konen import GPTEngine


class _FakeResponses:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response: SimpleNamespace) -> None:
        self.responses = _FakeResponses(response)


def _response(
    *,
    status: str | None = "completed",
    response_id: str | None = "resp_01HXYZ123",
    output_text: str = "The posterior probability remains above 0.75.",
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status, id=response_id, output_text=output_text, usage=usage
    )


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=128,
        output_tokens=42,
        total_tokens=170,
        input_tokens_details=SimpleNamespace(cached_tokens=64),
        output_tokens_details=SimpleNamespace(reasoning_tokens=18),
    )


def _engine(
    monkeypatch: pytest.MonkeyPatch,
    response: SimpleNamespace,
    config: dict[str, Any] | None = None,
) -> tuple[GPTEngine, _FakeClient]:
    client = _FakeClient(response)
    monkeypatch.setattr(GPTEngine, "_initialize_client", lambda self: client)
    engine = GPTEngine(
        model_family="gpt",
        model_name="gpt-5.6-luna",
        config={} if config is None else config,
        uid="gpt-analysis",
    )
    return engine, client


def test_gpt_engine_requires_gpt_family() -> None:
    with pytest.raises(ValueError, match="requires model_family='gpt'"):
        GPTEngine(model_family="gemini", model_name="gpt-5.6-luna")


def test_gpt_engine_client_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    client = object()

    def fake_openai(**kwargs: Any) -> object:
        calls.append(kwargs)
        return client

    monkeypatch.setattr(gpt_engine_module, "OpenAI", fake_openai)

    engine = GPTEngine(
        model_family="gpt", model_name="gpt-5.6-luna", uid="gpt-analysis"
    )

    assert engine.client is client
    assert calls == [{}]


def test_gpt_engine_client_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    client = object()

    def fake_openai(**kwargs: Any) -> object:
        calls.append(kwargs)
        return client

    monkeypatch.setattr(gpt_engine_module, "OpenAI", fake_openai)

    engine = GPTEngine(
        model_family="gpt",
        model_name="gpt-5.6-luna",
        config={"api_key": "test-api-key"},
        uid="gpt-analysis",
    )

    assert engine.client is client
    assert calls == [{"api_key": "test-api-key"}]


def test_gpt_engine_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, client = _engine(
        monkeypatch, _response(usage=_usage()), config={"reasoning_effort": "high"}
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

    assert client.responses.calls == [
        {
            "model": "gpt-5.6-luna",
            "input": (
                "Assess whether the posterior probability remains above 0.75 "
                "under the wider prior."
            ),
            "store": True,
            "instructions": (
                "Evaluate uncertainty using Bayesian inference and report assumptions."
            ),
            "reasoning": {"effort": "high"},
        }
    ]
    assert result.text == "The posterior probability remains above 0.75."
    assert result.metadata == {
        "engine_uid": "gpt-analysis",
        "model_family": "gpt",
        "model_name": "gpt-5.6-luna",
        "response_status": "completed",
        "provider_response_id": "resp_01HXYZ123",
        "token_usage": {
            "input_token_count": 128,
            "output_token_count": 42,
            "total_token_count": 170,
            "cached_token_count": 64,
            "reasoning_token_count": 18,
        },
    }
    assert result.provider_state == {"response_id": "resp_01HXYZ123"}


def test_gpt_engine_generate_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, client = _engine(monkeypatch, _response(response_id="resp_01HXYZ456"))

    result = engine.generate_response(
        prompt="Reassess the conclusion after widening the prior scale.",
        provider_state={"response_id": "resp_01HXYZ123"},
    )

    assert client.responses.calls == [
        {
            "model": "gpt-5.6-luna",
            "input": "Reassess the conclusion after widening the prior scale.",
            "store": True,
            "previous_response_id": "resp_01HXYZ123",
        }
    ]
    assert result.provider_state == {"response_id": "resp_01HXYZ456"}


def test_gpt_engine_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(
        monkeypatch,
        _response(
            status="incomplete",
            output_text="The posterior remains above the threshold, but",
        ),
    )

    result = engine.generate_response(prompt="Assess robustness under the wider prior.")

    assert result.text == "The posterior remains above the threshold, but"
    assert result.metadata["response_status"] == "incomplete"
    assert result.provider_state == {}


def test_gpt_engine_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(monkeypatch, _response(output_text=""))

    result = engine.generate_response(prompt="Assess robustness under the wider prior.")

    assert result.text is None


@pytest.mark.parametrize(
    "provider_state", [{}, {"response_id": ""}, {"response_id": 42}]
)
def test_gpt_engine_invalid_provider_state(
    monkeypatch: pytest.MonkeyPatch, provider_state: dict[str, Any]
) -> None:
    engine, _ = _engine(monkeypatch, _response())

    with pytest.raises(ValueError, match="non-empty 'response_id' string"):
        engine.generate_response(
            prompt="Assess robustness under the wider prior.",
            provider_state=provider_state,
        )


def test_gpt_engine_missing_status(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(monkeypatch, _response(status=None))

    with pytest.raises(RuntimeError, match="did not return a status"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_gpt_engine_terminal_status(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    engine, _ = _engine(monkeypatch, _response(status=status))

    with pytest.raises(RuntimeError, match=f"ended with status '{status}'"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


@pytest.mark.parametrize("status", ["queued", "in_progress"])
def test_gpt_engine_async_status(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    engine, _ = _engine(monkeypatch, _response(status=status))

    with pytest.raises(RuntimeError, match=f"status is '{status}'"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


def test_gpt_engine_unexpected_status(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(monkeypatch, _response(status="unknown"))

    with pytest.raises(RuntimeError, match="Unexpected OpenAI response status"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


@pytest.mark.parametrize("response_id", [None, ""])
def test_gpt_engine_completed_missing_id(
    monkeypatch: pytest.MonkeyPatch, response_id: str | None
) -> None:
    engine, _ = _engine(monkeypatch, _response(response_id=response_id))

    with pytest.raises(RuntimeError, match="did not return an ID"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")
