from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from konen import (
    AITeam,
    EngineRouter,
    LLMEngine,
    LLMPool,
    LLMResponse,
    Orchestrator,
    Workspace,
)


@dataclass(slots=True)
class _FakeEngine(LLMEngine):
    fail_once: bool = False
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

        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("Simulated provider failure.")

        return LLMResponse(
            text="The posterior probability remains above the decision threshold."
        )


def _workspace(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path / "orchestrator-error-study")
    workspace.initialize()
    workspace.add_agent(
        agent_uid="bayesian-analyst",
        name="Bayesian analyst",
        description="Evaluates uncertainty using Bayesian inference.",
    )
    return workspace


def _orchestrator(
    tmp_path: Path,
    *,
    engine: LLMEngine | None = None,
    router: EngineRouter | None = None,
    agent_uid: str | None = "bayesian-analyst",
) -> Orchestrator:
    workspace = _workspace(tmp_path)
    selected_engine = (
        _FakeEngine(
            model_family="gpt",
            model_name="gpt-5.6-luna",
            uid="gpt-analysis",
        )
        if engine is None
        else engine
    )
    return Orchestrator(
        workspace=workspace,
        ai_team=AITeam.from_workspace(workspace),
        llm_pool=LLMPool([selected_engine]),
        engine_router=(
            EngineRouter({"bayesian-analyst": "gpt-analysis"})
            if router is None
            else router
        ),
        agent_uid=agent_uid,
    )


def test_orchestrator_engine_uid_no_agent() -> None:
    orchestrator = Orchestrator()

    with pytest.raises(ValueError, match="agent_uid must be set"):
        _ = orchestrator.engine_uid


def test_orchestrator_run_turn_no_agent(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, agent_uid=None)

    with pytest.raises(ValueError, match="agent_uid must be set before running"):
        orchestrator.run_turn(
            "Estimate the posterior probability under the baseline prior."
        )


def test_orchestrator_missing_agent(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.agent_uid = "causal-analyst"

    with pytest.raises(ValueError, match="does not exist in AI team"):
        orchestrator.run_turn("Assess whether the effect is causally identifiable.")


def test_orchestrator_missing_mapping(tmp_path: Path) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        router=EngineRouter(),
    )

    with pytest.raises(ValueError, match="No engine mapping exists"):
        orchestrator.run_turn(
            "Estimate the posterior probability under the baseline prior."
        )


def test_orchestrator_missing_engine(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    orchestrator = Orchestrator(
        workspace=workspace,
        ai_team=AITeam.from_workspace(workspace),
        llm_pool=LLMPool(),
        engine_router=EngineRouter(
            {
                "bayesian-analyst": "gpt-analysis",
            }
        ),
        agent_uid="bayesian-analyst",
    )

    with pytest.raises(ValueError, match="does not exist in LLM pool"):
        orchestrator.run_turn(
            "Estimate the posterior probability under the baseline prior."
        )


def test_orchestrator_invalid_history_limit(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)

    with pytest.raises(ValueError, match="history_limit must be non-negative"):
        orchestrator.run_turn(
            "Estimate the posterior probability under the baseline prior.",
            history_limit=-1,
        )


def test_orchestrator_from_workspace_missing_agent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="does not exist in AI team"):
        Orchestrator.from_workspace(
            workspace,
            agent_uid="causal-analyst",
        )


def test_orchestrator_reload_removed_agent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    orchestrator = Orchestrator(
        workspace=workspace,
        ai_team=AITeam.from_workspace(workspace),
        agent_uid="bayesian-analyst",
    )

    workspace.remove_agent("bayesian-analyst")
    orchestrator.reload_from_workspace()

    assert orchestrator.agent_uid is None
    assert orchestrator.ai_team.list_agents() == []


def test_orchestrator_uninitialized_workspace(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        workspace=Workspace(tmp_path / "uninitialized-study"),
        agent_uid="bayesian-analyst",
    )

    with pytest.raises(RuntimeError, match="Workspace is not initialized"):
        orchestrator.run_turn(
            "Estimate the posterior probability under the baseline prior."
        )


def test_orchestrator_incomplete_round(tmp_path: Path) -> None:
    engine = _FakeEngine(
        model_family="gpt",
        model_name="gpt-5.6-luna",
        uid="gpt-analysis",
        fail_once=True,
    )
    orchestrator = _orchestrator(tmp_path, engine=engine)

    with pytest.raises(RuntimeError, match="Simulated provider failure"):
        orchestrator.run_turn(
            "Estimate the posterior probability under the baseline prior."
        )

    current_round = orchestrator.interaction_state.round
    assert current_round is not None
    assert current_round.transcript.completed_at is None
    assert orchestrator.interaction_state.transcript_history.rounds == []

    with pytest.raises(RuntimeError, match="current round is incomplete"):
        orchestrator.run_turn(
            "Reassess the posterior probability under the wider prior."
        )

    assert len(engine.calls) == 1

    orchestrator.interaction_state.clear_round()
    result = orchestrator.run_turn(
        "Reassess the posterior probability under the wider prior."
    )

    assert result == ("The posterior probability remains above the decision threshold.")
    assert len(engine.calls) == 2
    assert len(orchestrator.interaction_state.transcript_history.rounds) == 1
