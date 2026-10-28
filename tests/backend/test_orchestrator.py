from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from konen import (
    EngineFactory,
    LLMEngine,
    LLMResponse,
    Orchestrator,
    Workspace,
)


@dataclass(slots=True)
class _FakeEngine(LLMEngine):
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)

    def _initialize_client(self) -> object:
        return object()

    def generate_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        provider_state: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "provider_state": provider_state,
            }
        )

        if self.uid == "gemini-analysis":
            outputs = [
                "Action A has the higher expected utility.",
                "Action A remains preferred after the sensitivity analysis.",
            ]
        else:
            outputs = [
                "The baseline posterior probability is 0.82.",
                "The wider prior lowers the posterior probability to 0.79.",
                "The posterior remains above the 0.75 decision threshold.",
                "The conclusion is stable across the tested prior scales.",
            ]

        index = min(len(self.calls) - 1, len(outputs) - 1)
        return LLMResponse(text=outputs[index])


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


def _workspace(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path / "bayesian-decision-study")
    workspace.initialize()
    workspace.add_agent(
        agent_uid="bayesian-analyst",
        name="Bayesian analyst",
        description=(
            "Evaluates uncertainty using Bayesian inference and posterior reasoning."
        ),
    )
    workspace.add_agent(
        agent_uid="decision-analyst",
        name="Decision analyst",
        description=(
            "Compares actions using expected utility and decision-theoretic trade-offs."
        ),
    )
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
    workspace.assign_engine("bayesian-analyst", "gpt-analysis")
    workspace.assign_engine("decision-analyst", "gemini-analysis")
    workspace.append_to_memory(
        "research/prior_notes.md",
        "The baseline prior odds favor the first hypothesis by 2:1.\n",
    )
    workspace.append_to_scratchpad(
        "bayesian-analyst",
        "Compare prior scales of 1 and 2 before reporting the conclusion.\n",
    )
    workspace.append_to_scratchpad(
        "decision-analyst",
        "Compare expected utility under the baseline and sensitivity scenarios.\n",
    )
    return workspace


def _orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent_uid: str = "bayesian-analyst",
) -> Orchestrator:
    monkeypatch.setattr(
        EngineFactory,
        "build",
        staticmethod(_fake_build),
    )
    return Orchestrator.from_workspace(
        workspace=_workspace(tmp_path),
        agent_uid=agent_uid,
    )


def test_orchestrator_from_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(tmp_path, monkeypatch)

    assert orchestrator.ai_team.list_agents() == [
        "bayesian-analyst",
        "decision-analyst",
    ]
    assert orchestrator.llm_pool.list_engines() == [
        "gemini-analysis",
        "gpt-analysis",
    ]
    assert orchestrator.engine_router.table == {
        "bayesian-analyst": "gpt-analysis",
        "decision-analyst": "gemini-analysis",
    }
    assert orchestrator.agent_uid == "bayesian-analyst"


def test_orchestrator_engine_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(tmp_path, monkeypatch)

    assert orchestrator.engine_uid == "gpt-analysis"

    orchestrator.agent_uid = "decision-analyst"

    assert orchestrator.engine_uid == "gemini-analysis"


def test_orchestrator_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    orchestrator.workspace.assign_engine(
        "bayesian-analyst",
        "gemini-analysis",
    )

    orchestrator.reload_from_workspace()

    assert orchestrator.agent_uid == "bayesian-analyst"
    assert orchestrator.engine_uid == "gemini-analysis"
    assert orchestrator.llm_pool.get_engine("gemini-analysis").model_name == (
        "gemini-3.5-flash-lite"
    )


def test_orchestrator_first_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    engine = orchestrator.llm_pool.get_engine("gpt-analysis")
    assert isinstance(engine, _FakeEngine)

    result = orchestrator.run_turn(
        "Estimate the posterior probability under the baseline prior."
    )

    assert result == "The baseline posterior probability is 0.82."
    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert call["provider_state"] is None
    assert "# Transcript history" not in call["prompt"]
    assert (
        "Estimate the posterior probability under the baseline prior."
        in (call["prompt"])
    )
    assert "Agent identifier: bayesian-analyst" in call["system_instruction"]
    assert "Agent identifier: decision-analyst" in call["system_instruction"]
    assert "memory/research/prior_notes.md" in call["system_instruction"]
    assert (
        "The baseline prior odds favor the first hypothesis by 2:1."
        in (call["system_instruction"])
    )
    assert "scratchpad/bayesian-analyst.md" in call["system_instruction"]
    assert "Compare prior scales of 1 and 2" in call["system_instruction"]


def test_orchestrator_selected_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    orchestrator.workspace.append_to_memory(
        "research/posterior_summary.md",
        "The posterior probability is 0.82.\n",
    )
    engine = orchestrator.llm_pool.get_engine("gpt-analysis")
    assert isinstance(engine, _FakeEngine)

    orchestrator.run_turn(
        "Compare the two findings.",
        memory_filenames=("research/posterior_summary.md",),
        scratchpad_agent_uids=("decision-analyst",),
    )

    interaction_round = orchestrator.interaction_state.round
    assert interaction_round is not None
    assert interaction_round.memory_context == {
        "memory/research/posterior_summary.md": "The posterior probability is 0.82.\n"
    }
    assert interaction_round.scratchpad_context == {
        "scratchpad/decision-analyst.md": (
            "Compare expected utility under the baseline and sensitivity scenarios.\n"
        )
    }
    system_instruction = engine.calls[0]["system_instruction"]
    assert "The posterior probability is 0.82." in system_instruction
    assert "Compare expected utility under the baseline" in system_instruction
    assert "The baseline prior odds favor" not in system_instruction
    assert "Compare prior scales of 1 and 2" not in system_instruction


def test_orchestrator_empty_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    engine = orchestrator.llm_pool.get_engine("gpt-analysis")
    assert isinstance(engine, _FakeEngine)

    orchestrator.run_turn(
        "Assess the evidence without workspace notes.",
        memory_filenames=(),
        scratchpad_agent_uids=(),
    )

    interaction_round = orchestrator.interaction_state.round
    assert interaction_round is not None
    assert interaction_round.memory_context == {}
    assert interaction_round.scratchpad_context == {}
    system_instruction = engine.calls[0]["system_instruction"]
    assert "## Memory context" not in system_instruction
    assert "## Scratchpad context" not in system_instruction


def test_orchestrator_selected_memory_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    orchestrator.workspace.remove_memory_file("research/prior_notes.md")

    with pytest.raises(FileNotFoundError, match="Memory file does not exist"):
        orchestrator.run_turn(
            "Reassess the prior.",
            memory_filenames=("research/prior_notes.md",),
        )

    assert orchestrator.interaction_state.round is None


def test_orchestrator_second_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    engine = orchestrator.llm_pool.get_engine("gpt-analysis")
    assert isinstance(engine, _FakeEngine)

    orchestrator.run_turn(
        "Estimate the posterior probability under the baseline prior."
    )
    result = orchestrator.run_turn(
        "Reassess the posterior probability under the wider prior."
    )

    assert result == "The wider prior lowers the posterior probability to 0.79."
    prompt = engine.calls[1]["prompt"]
    assert "# Transcript history" in prompt
    assert "Estimate the posterior probability under the baseline prior." in prompt
    assert "The baseline posterior probability is 0.82." in prompt
    assert "Reassess the posterior probability under the wider prior." in prompt


def test_orchestrator_history_limit_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    engine = orchestrator.llm_pool.get_engine("gpt-analysis")
    assert isinstance(engine, _FakeEngine)

    orchestrator.run_turn(
        "Estimate the posterior probability under the baseline prior."
    )
    orchestrator.run_turn(
        "Reassess the posterior probability without transcript history.",
        history_limit=0,
    )

    prompt = engine.calls[1]["prompt"]
    assert "# Transcript history" not in prompt
    assert "Estimate the posterior probability under the baseline prior." not in prompt
    assert "The baseline posterior probability is 0.82." not in prompt
    assert "Reassess the posterior probability without transcript history." in prompt


def test_orchestrator_history_limit_finite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    engine = orchestrator.llm_pool.get_engine("gpt-analysis")
    assert isinstance(engine, _FakeEngine)

    orchestrator.run_turn(
        "Estimate the posterior probability under the baseline prior."
    )
    orchestrator.run_turn("Reassess the posterior probability under the wider prior.")
    orchestrator.run_turn(
        "Check whether the posterior remains above the decision threshold.",
        history_limit=1,
    )

    prompt = engine.calls[2]["prompt"]
    assert "Estimate the posterior probability under the baseline prior." not in prompt
    assert "The baseline posterior probability is 0.82." not in prompt
    assert "Reassess the posterior probability under the wider prior." in prompt
    assert "The wider prior lowers the posterior probability to 0.79." in prompt
    assert "Check whether the posterior remains above the decision threshold." in prompt


def test_orchestrator_history_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    engine = orchestrator.llm_pool.get_engine("gpt-analysis")
    assert isinstance(engine, _FakeEngine)

    orchestrator.run_turn(
        "Estimate the posterior probability under the baseline prior."
    )
    orchestrator.run_turn("Reassess the posterior probability under the wider prior.")
    orchestrator.run_turn(
        "Check whether the posterior remains above the decision threshold."
    )
    orchestrator.run_turn(
        "Summarize the sensitivity analysis.",
        history_limit=None,
    )

    prompt = engine.calls[3]["prompt"]
    assert "Estimate the posterior probability under the baseline prior." in prompt
    assert "The baseline posterior probability is 0.82." in prompt
    assert "Reassess the posterior probability under the wider prior." in prompt
    assert "The wider prior lowers the posterior probability to 0.79." in prompt
    assert "Check whether the posterior remains above the decision threshold." in prompt
    assert "The posterior remains above the 0.75 decision threshold." in prompt
    assert "Summarize the sensitivity analysis." in prompt


def test_orchestrator_routes_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        monkeypatch,
        agent_uid="decision-analyst",
    )
    gpt = orchestrator.llm_pool.get_engine("gpt-analysis")
    gemini = orchestrator.llm_pool.get_engine("gemini-analysis")
    assert isinstance(gpt, _FakeEngine)
    assert isinstance(gemini, _FakeEngine)

    result = orchestrator.run_turn(
        "Compare the expected utility of the available actions."
    )

    assert result == "Action A has the higher expected utility."
    assert gpt.calls == []
    assert len(gemini.calls) == 1
    assert (
        "Your agent identifier is: decision-analyst"
        in (gemini.calls[0]["system_instruction"])
    )
    assert (
        "Compare the expected utility of the available actions."
        in (gemini.calls[0]["prompt"])
    )
