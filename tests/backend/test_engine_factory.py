from typing import Any

import pytest

from konen import EngineFactory, LLMEngine, LLMResponse


class _FakeEngine(LLMEngine):
    def _initialize_client(self) -> object:
        return object()

    def generate_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        provider_state: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text="The posterior probability remains above the decision threshold.",
            metadata=self._build_response_metadata(),
        )


def _fake_builder(kwargs: dict[str, Any]) -> LLMEngine:
    return _FakeEngine(**kwargs)


def test_engine_factory_default_families() -> None:
    assert set(EngineFactory._builders) == {"claude", "gemini", "gpt", "kimi"}


def test_engine_factory_register(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineFactory, "_builders", dict(EngineFactory._builders))

    EngineFactory.register("research", _fake_builder)

    assert EngineFactory._builders["research"] is _fake_builder


def test_engine_factory_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineFactory, "_builders", dict(EngineFactory._builders))

    def first_builder(kwargs: dict[str, Any]) -> LLMEngine:
        return _FakeEngine(**kwargs)

    def second_builder(kwargs: dict[str, Any]) -> LLMEngine:
        return _FakeEngine(**kwargs)

    EngineFactory.register("research", first_builder)
    EngineFactory.register("research", second_builder)

    assert EngineFactory._builders["research"] is second_builder


def test_engine_factory_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineFactory, "_builders", dict(EngineFactory._builders))
    EngineFactory.register("research", _fake_builder)

    engine = EngineFactory.build(
        model_family="research",
        model_name="research-model-v1",
        config={"reasoning_effort": "high"},
        uid="research-analysis",
    )

    assert isinstance(engine, _FakeEngine)
    assert engine.model_family == "research"
    assert engine.model_name == "research-model-v1"
    assert engine.config == {"reasoning_effort": "high"}
    assert engine.uid == "research-analysis"


def test_engine_factory_build_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineFactory, "_builders", dict(EngineFactory._builders))
    EngineFactory.register("research", _fake_builder)

    engine = EngineFactory.build(
        model_family="research", model_name="research-model-v1"
    )

    assert engine.model_family == "research"
    assert engine.model_name == "research-model-v1"
    assert engine.config == {}


def test_engine_factory_copies_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EngineFactory, "_builders", dict(EngineFactory._builders))
    captured: dict[str, Any] = {}

    def builder(kwargs: dict[str, Any]) -> LLMEngine:
        captured.update(kwargs)
        return _FakeEngine(**kwargs)

    EngineFactory.register("research", builder)
    config = {"reasoning_effort": "high"}

    EngineFactory.build(
        model_family="research",
        model_name="research-model-v1",
        config=config,
        uid="research-analysis",
    )

    assert captured["config"] == {"reasoning_effort": "high"}
    assert captured["config"] is not config


def test_engine_factory_unsupported_family() -> None:
    with pytest.raises(ValueError, match="Unsupported model_family: 'research'"):
        EngineFactory.build(model_family="research", model_name="research-model-v1")


def test_engine_factory_validate_kwargs() -> None:
    EngineFactory._validate_builder_kwargs(
        {
            "model_family": "gpt",
            "model_name": "gpt-5.6-luna",
            "config": {"reasoning_effort": "high"},
            "uid": "gpt-analysis",
        }
    )


def test_engine_factory_invalid_kwargs() -> None:
    with pytest.raises(ValueError, match="Unexpected engine kwargs: api_key"):
        EngineFactory._validate_builder_kwargs(
            {"model_family": "gpt", "model_name": "gpt-5.6-luna", "api_key": "not-used"}
        )
