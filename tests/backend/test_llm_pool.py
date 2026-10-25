from pathlib import Path
from typing import Any

import pytest

from konen import EngineFactory, LLMEngine, LLMPool, LLMResponse, Workspace


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


def _engines() -> list[LLMEngine]:
    return [
        _FakeEngine(
            model_family="gpt",
            model_name="gpt-5.6-luna",
            config={"reasoning_effort": "high"},
            uid="gpt-analysis",
        ),
        _FakeEngine(
            model_family="gemini",
            model_name="gemini-3.5-flash-lite",
            uid="gemini-analysis",
        ),
    ]


def _fake_build(
    model_family: str,
    model_name: str,
    config: dict[str, Any] | None = None,
    uid: str | None = None,
) -> LLMEngine:
    return _FakeEngine(
        model_family=model_family,
        model_name=model_name,
        config={} if config is None else config,
        uid="generated-engine" if uid is None else uid,
    )


def test_llm_pool_defaults() -> None:
    pool = LLMPool()

    assert pool.list_engines() == []


def test_llm_pool_engines() -> None:
    pool = LLMPool(reversed(_engines()))

    assert pool.list_engines() == ["gemini-analysis", "gpt-analysis"]
    assert pool.get_engine("gpt-analysis").model_name == "gpt-5.6-luna"
    assert pool.get_engine("gemini-analysis").model_name == "gemini-3.5-flash-lite"


def test_llm_pool_duplicate_engine() -> None:
    engine = _engines()[0]

    with pytest.raises(ValueError, match="already exists in LLM pool"):
        LLMPool([engine, engine])


def test_llm_pool_missing_engine() -> None:
    pool = LLMPool(_engines())

    with pytest.raises(ValueError, match="does not exist in LLM pool"):
        pool.get_engine("claude-analysis")


def test_llm_pool_add_remove() -> None:
    pool = LLMPool()
    engine = _FakeEngine(
        model_family="claude", model_name="claude-sonnet-4-5", uid="claude-analysis"
    )

    pool.add_engine(engine)

    assert pool.get_engine("claude-analysis") is engine

    pool.remove_engine("claude-analysis")

    assert pool.list_engines() == []


def test_llm_pool_add_duplicate() -> None:
    engine = _engines()[0]
    pool = LLMPool([engine])

    with pytest.raises(ValueError, match="already exists in LLM pool"):
        pool.add_engine(engine)


def test_llm_pool_remove_missing() -> None:
    pool = LLMPool(_engines())

    with pytest.raises(ValueError, match="does not exist in LLM pool"):
        pool.remove_engine("claude-analysis")


def test_llm_pool_from_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace(tmp_path / "bayesian-model-study")
    workspace.initialize()
    workspace.add_engine(
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        config={"reasoning_effort": "high"},
    )
    workspace.add_engine(
        engine_uid="gemini-analysis",
        model_family="gemini",
        model_name="gemini-3.5-flash-lite",
    )
    monkeypatch.setattr(EngineFactory, "build", staticmethod(_fake_build))

    pool = LLMPool.from_workspace(workspace)

    assert pool.list_engines() == ["gemini-analysis", "gpt-analysis"]
    gpt = pool.get_engine("gpt-analysis")
    assert gpt.model_family == "gpt"
    assert gpt.model_name == "gpt-5.6-luna"
    assert gpt.config == {"reasoning_effort": "high"}


def test_llm_pool_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path / "bayesian-model-study")
    workspace.initialize()
    workspace.add_engine(
        engine_uid="gemini-analysis",
        model_family="gemini",
        model_name="gemini-3.5-flash-lite",
    )
    monkeypatch.setattr(EngineFactory, "build", staticmethod(_fake_build))
    pool = LLMPool(
        [_FakeEngine(model_family="gpt", model_name="gpt-5.6-luna", uid="gpt-analysis")]
    )

    pool.reload_from_workspace(workspace)

    assert pool.list_engines() == ["gemini-analysis"]
    assert pool.get_engine("gemini-analysis").model_name == "gemini-3.5-flash-lite"
