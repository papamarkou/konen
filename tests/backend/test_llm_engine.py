from typing import Any
from uuid import UUID

import pytest

from konen import LLMEngine, LLMResponse


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
            provider_state={} if provider_state is None else provider_state,
        )


def test_llm_engine_is_abstract() -> None:
    with pytest.raises(TypeError):
        LLMEngine(model_family="gpt", model_name="gpt-5.6-luna")


def test_llm_engine_initializes() -> None:
    engine = _FakeEngine(
        model_family="gpt",
        model_name="gpt-5.6-luna",
        config={"reasoning_effort": "high"},
        uid="gpt-analysis",
    )

    assert engine.model_family == "gpt"
    assert engine.model_name == "gpt-5.6-luna"
    assert engine.config == {"reasoning_effort": "high"}
    assert engine.uid == "gpt-analysis"
    assert engine.client is not None


def test_llm_engine_default_uid() -> None:
    engine = _FakeEngine(model_family="gpt", model_name="gpt-5.6-luna")

    assert UUID(engine.uid).version == 4


def test_llm_engine_copies_config() -> None:
    config = {"reasoning_effort": "high"}
    engine = _FakeEngine(
        model_family="gpt", model_name="gpt-5.6-luna", config=config, uid="gpt-analysis"
    )

    config["reasoning_effort"] = "medium"

    assert engine.config == {"reasoning_effort": "high"}


def test_llm_engine_provider_state_empty() -> None:
    engine = _FakeEngine(
        model_family="gpt", model_name="gpt-5.6-luna", uid="gpt-analysis"
    )

    assert engine.prepare_provider_state([]) is None


def test_llm_engine_provider_state() -> None:
    engine = _FakeEngine(
        model_family="gpt", model_name="gpt-5.6-luna", uid="gpt-analysis"
    )
    states = [
        {"response": {"id": "resp_01HXYZ123", "status": "completed"}},
        {"response": {"id": "resp_01HXYZ456", "status": "completed"}},
    ]

    prepared = engine.prepare_provider_state(states)

    assert prepared == states[-1]
    assert prepared is not states[-1]
    prepared["response"]["status"] = "changed"
    assert states[-1]["response"]["status"] == "completed"


def test_llm_engine_response_metadata() -> None:
    engine = _FakeEngine(
        model_family="gpt", model_name="gpt-5.6-luna", uid="gpt-analysis"
    )

    assert engine._build_response_metadata() == {
        "engine_uid": "gpt-analysis",
        "model_family": "gpt",
        "model_name": "gpt-5.6-luna",
    }
