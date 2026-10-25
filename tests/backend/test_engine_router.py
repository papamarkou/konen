from pathlib import Path

import pytest

from konen import EngineRouter, Workspace


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
    return workspace


def test_engine_router_defaults() -> None:
    router = EngineRouter()

    assert router.table == {}


def test_engine_router_copies_table() -> None:
    table = {"bayesian-analyst": "gpt-analysis", "decision-analyst": "gemini-analysis"}
    router = EngineRouter(table=table)

    table["bayesian-analyst"] = "gemini-analysis"

    assert router.table == {
        "bayesian-analyst": "gpt-analysis",
        "decision-analyst": "gemini-analysis",
    }


def test_engine_router_assign() -> None:
    router = EngineRouter(table={"bayesian-analyst": "gpt-analysis"})

    router.assign_engine("decision-analyst", "gemini-analysis")
    router.assign_engine("bayesian-analyst", "gemini-analysis")

    assert router.table == {
        "bayesian-analyst": "gemini-analysis",
        "decision-analyst": "gemini-analysis",
    }


def test_engine_router_get() -> None:
    router = EngineRouter(table={"bayesian-analyst": "gpt-analysis"})

    assert router.get_engine_uid("bayesian-analyst") == "gpt-analysis"


def test_engine_router_get_missing() -> None:
    router = EngineRouter()

    with pytest.raises(ValueError, match="No engine mapping exists"):
        router.get_engine_uid("causal-analyst")


def test_engine_router_unassign() -> None:
    router = EngineRouter(
        table={
            "bayesian-analyst": "gpt-analysis",
            "decision-analyst": "gemini-analysis",
        }
    )

    router.unassign_engine("bayesian-analyst")

    assert router.table == {"decision-analyst": "gemini-analysis"}


def test_engine_router_unassign_missing() -> None:
    router = EngineRouter()

    with pytest.raises(ValueError, match="No engine mapping exists"):
        router.unassign_engine("causal-analyst")


def test_engine_router_from_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    router = EngineRouter.from_workspace(workspace)

    assert router.table == {
        "bayesian-analyst": "gpt-analysis",
        "decision-analyst": "gemini-analysis",
    }


def test_engine_router_reload(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    router = EngineRouter(table={"bayesian-analyst": "gemini-analysis"})
    workspace.assign_engine("bayesian-analyst", "gemini-analysis")

    router.reload_from_workspace(workspace)

    assert router.table == {
        "bayesian-analyst": "gemini-analysis",
        "decision-analyst": "gemini-analysis",
    }
