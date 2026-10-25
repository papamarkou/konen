from types import SimpleNamespace
from typing import Any

import pytest

import konen.backend.gemini_engine as gemini_engine_module
from konen import GeminiEngine


class _FakeInteractions:
    def __init__(self, interaction: Any) -> None:
        self.interaction = interaction
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.interaction


class _FakeClient:
    def __init__(self, interaction: Any) -> None:
        self.interactions = _FakeInteractions(interaction)


class _Interaction:
    def __init__(
        self,
        *,
        status: Any = "completed",
        interaction_id: str | None = "v1_interaction_01",
        output_text: str | None = ("The posterior probability remains above 0.75."),
        usage: SimpleNamespace | None = None,
        output_error: Exception | None = None,
    ) -> None:
        self.status = status
        self.id = interaction_id
        self.usage = usage
        self._output_text = output_text
        self._output_error = output_error

    @property
    def output_text(self) -> str | None:
        if self._output_error is not None:
            raise self._output_error
        return self._output_text


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        total_input_tokens=128,
        total_output_tokens=42,
        total_tokens=196,
        total_cached_tokens=64,
        total_thought_tokens=24,
        total_tool_use_tokens=6,
    )


def _engine(
    monkeypatch: pytest.MonkeyPatch,
    interaction: Any,
    config: dict[str, Any] | None = None,
) -> tuple[GeminiEngine, _FakeClient]:
    client = _FakeClient(interaction)
    monkeypatch.setattr(GeminiEngine, "_initialize_client", lambda self: client)
    engine = GeminiEngine(
        model_family="gemini",
        model_name="gemini-3.5-flash-lite",
        config={} if config is None else config,
        uid="gemini-analysis",
    )
    return engine, client


def test_gemini_engine_requires_gemini_family() -> None:
    with pytest.raises(ValueError, match="requires model_family='gemini'"):
        GeminiEngine(model_family="gpt", model_name="gemini-3.5-flash-lite")


def test_gemini_engine_client_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    client = object()

    def fake_client(**kwargs: Any) -> object:
        calls.append(kwargs)
        return client

    monkeypatch.setattr(gemini_engine_module.genai, "Client", fake_client)

    engine = GeminiEngine(
        model_family="gemini", model_name="gemini-3.5-flash-lite", uid="gemini-analysis"
    )

    assert engine.client is client
    assert calls == [{}]


def test_gemini_engine_client_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    client = object()

    def fake_client(**kwargs: Any) -> object:
        calls.append(kwargs)
        return client

    monkeypatch.setattr(gemini_engine_module.genai, "Client", fake_client)

    engine = GeminiEngine(
        model_family="gemini",
        model_name="gemini-3.5-flash-lite",
        config={"api_key": "test-api-key"},
        uid="gemini-analysis",
    )

    assert engine.client is client
    assert calls == [{"api_key": "test-api-key"}]


def test_gemini_engine_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, client = _engine(
        monkeypatch,
        _Interaction(status=SimpleNamespace(value="completed"), usage=_usage()),
        config={"thinking_level": "high"},
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

    assert client.interactions.calls == [
        {
            "model": "gemini-3.5-flash-lite",
            "input": (
                "Assess whether the posterior probability remains above 0.75 "
                "under the wider prior."
            ),
            "store": True,
            "system_instruction": (
                "Evaluate uncertainty using Bayesian inference and report assumptions."
            ),
            "generation_config": {"thinking_level": "high"},
        }
    ]
    assert result.text == "The posterior probability remains above 0.75."
    assert result.metadata == {
        "engine_uid": "gemini-analysis",
        "model_family": "gemini",
        "model_name": "gemini-3.5-flash-lite",
        "interaction_status": "completed",
        "provider_response_id": "v1_interaction_01",
        "token_usage": {
            "input_token_count": 128,
            "output_token_count": 42,
            "total_token_count": 196,
            "cached_token_count": 64,
            "thought_token_count": 24,
            "tool_use_token_count": 6,
        },
    }
    assert result.provider_state == {"interaction_id": "v1_interaction_01"}


def test_gemini_engine_generate_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, client = _engine(
        monkeypatch, _Interaction(interaction_id="v1_interaction_02")
    )

    result = engine.generate_response(
        prompt="Reassess the conclusion after widening the prior scale.",
        provider_state={"interaction_id": "v1_interaction_01"},
    )

    assert client.interactions.calls == [
        {
            "model": "gemini-3.5-flash-lite",
            "input": "Reassess the conclusion after widening the prior scale.",
            "store": True,
            "previous_interaction_id": "v1_interaction_01",
        }
    ]
    assert result.provider_state == {"interaction_id": "v1_interaction_02"}


def test_gemini_engine_usage_omits_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = SimpleNamespace(
        total_input_tokens=128,
        total_output_tokens=42,
        total_tokens=170,
        total_cached_tokens=None,
        total_thought_tokens=None,
        total_tool_use_tokens=None,
    )
    engine, _ = _engine(monkeypatch, _Interaction(usage=usage))

    result = engine.generate_response(prompt="Assess robustness under the wider prior.")

    assert result.metadata["token_usage"] == {
        "input_token_count": 128,
        "output_token_count": 42,
        "total_token_count": 170,
    }


def test_gemini_engine_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(
        monkeypatch,
        _Interaction(
            status="incomplete",
            output_text="The posterior remains above the threshold, but",
        ),
    )

    result = engine.generate_response(prompt="Assess robustness under the wider prior.")

    assert result.text == "The posterior remains above the threshold, but"
    assert result.metadata["interaction_status"] == "incomplete"
    assert result.provider_state == {}


@pytest.mark.parametrize(
    "output_error", [ValueError("no text output"), AttributeError("output_text")]
)
def test_gemini_engine_missing_output(
    monkeypatch: pytest.MonkeyPatch, output_error: Exception
) -> None:
    engine, _ = _engine(monkeypatch, _Interaction(output_error=output_error))

    result = engine.generate_response(prompt="Assess robustness under the wider prior.")

    assert result.text is None


@pytest.mark.parametrize(
    "provider_state", [{}, {"interaction_id": ""}, {"interaction_id": 42}]
)
def test_gemini_engine_invalid_provider_state(
    monkeypatch: pytest.MonkeyPatch, provider_state: dict[str, Any]
) -> None:
    engine, _ = _engine(monkeypatch, _Interaction())

    with pytest.raises(ValueError, match="non-empty 'interaction_id' string"):
        engine.generate_response(
            prompt="Assess robustness under the wider prior.",
            provider_state=provider_state,
        )


def test_gemini_engine_missing_status(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(monkeypatch, _Interaction(status=None))

    with pytest.raises(RuntimeError, match="did not return a status"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


@pytest.mark.parametrize("status", ["failed", "cancelled", "budget_exceeded"])
def test_gemini_engine_terminal_status(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    engine, _ = _engine(monkeypatch, _Interaction(status=status))

    with pytest.raises(RuntimeError, match=f"ended with status '{status}'"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


@pytest.mark.parametrize("status", ["queued", "in_progress", "requires_action"])
def test_gemini_engine_async_status(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    engine, _ = _engine(monkeypatch, _Interaction(status=status))

    with pytest.raises(RuntimeError, match=f"status is '{status}'"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


def test_gemini_engine_unexpected_status(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _engine(monkeypatch, _Interaction(status="unknown"))

    with pytest.raises(RuntimeError, match="Unexpected Gemini interaction status"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")


@pytest.mark.parametrize("interaction_id", [None, ""])
def test_gemini_engine_completed_missing_id(
    monkeypatch: pytest.MonkeyPatch, interaction_id: str | None
) -> None:
    engine, _ = _engine(monkeypatch, _Interaction(interaction_id=interaction_id))

    with pytest.raises(RuntimeError, match="did not return an ID"):
        engine.generate_response(prompt="Assess robustness under the wider prior.")
