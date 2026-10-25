import json
from pathlib import Path

import pytest

from konen import AIAgent, AITeam, Workspace


def _agents() -> list[AIAgent]:
    return [
        AIAgent(
            name="Bayesian analyst",
            description=(
                "Evaluates uncertainty using Bayesian inference and posterior reasoning."
            ),
            uid="bayesian-analyst",
        ),
        AIAgent(
            name="Decision analyst",
            description=(
                "Compares actions using expected utility and decision-theoretic trade-offs."
            ),
            uid="decision-analyst",
        ),
    ]


def test_ai_team_defaults() -> None:
    team = AITeam()

    assert team.list_agents() == []
    assert team.to_context() == {}


def test_ai_team_agents() -> None:
    team = AITeam(reversed(_agents()))

    assert team.list_agents() == ["bayesian-analyst", "decision-analyst"]
    assert team.get_agent("bayesian-analyst").name == "Bayesian analyst"
    assert team.get_agent("decision-analyst").name == "Decision analyst"


def test_ai_team_duplicate_agent() -> None:
    agent = AIAgent(
        name="Bayesian analyst",
        description=(
            "Evaluates uncertainty using Bayesian inference and posterior reasoning."
        ),
        uid="bayesian-analyst",
    )

    with pytest.raises(ValueError, match="already exists in AI team"):
        AITeam([agent, agent])


def test_ai_team_missing_agent() -> None:
    team = AITeam(_agents())

    with pytest.raises(ValueError, match="does not exist in AI team"):
        team.get_agent("causal-analyst")


def test_ai_team_to_context() -> None:
    team = AITeam(reversed(_agents()))

    assert team.to_context() == {
        "bayesian-analyst": {
            "name": "Bayesian analyst",
            "description": (
                "Evaluates uncertainty using Bayesian inference and posterior reasoning."
            ),
        },
        "decision-analyst": {
            "name": "Decision analyst",
            "description": (
                "Compares actions using expected utility and decision-theoretic trade-offs."
            ),
        },
    }


def test_ai_team_add_remove() -> None:
    team = AITeam()
    agent = AIAgent(
        name="Causal analyst",
        description="Assesses causal identification and treatment-effect assumptions.",
        uid="causal-analyst",
    )

    team.add_agent(agent)

    assert team.get_agent("causal-analyst") is agent

    team.remove_agent("causal-analyst")

    assert team.list_agents() == []


def test_ai_team_add_duplicate() -> None:
    agent = _agents()[0]
    team = AITeam([agent])

    with pytest.raises(ValueError, match="already exists in AI team"):
        team.add_agent(agent)


def test_ai_team_remove_missing() -> None:
    team = AITeam(_agents())

    with pytest.raises(ValueError, match="does not exist in AI team"):
        team.remove_agent("causal-analyst")


def test_ai_team_from_workspace(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-decision-study")
    workspace.initialize()
    workspace.agents_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "agents": {
                    "bayesian-analyst": {
                        "name": "Bayesian analyst",
                        "description": (
                            "Evaluates uncertainty using Bayesian inference and "
                            "posterior reasoning."
                        ),
                    },
                    "decision-analyst": {
                        "name": "Decision analyst",
                        "description": (
                            "Compares actions using expected utility and "
                            "decision-theoretic trade-offs."
                        ),
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    team = AITeam.from_workspace(workspace)

    assert team.list_agents() == ["bayesian-analyst", "decision-analyst"]
    assert team.get_agent("bayesian-analyst").description == (
        "Evaluates uncertainty using Bayesian inference and posterior reasoning."
    )


def test_ai_team_reload(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-decision-study")
    workspace.initialize()
    workspace.agents_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "agents": {
                    "causal-analyst": {
                        "name": "Causal analyst",
                        "description": (
                            "Assesses causal identification and treatment-effect assumptions."
                        ),
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    team = AITeam(_agents())

    team.reload_from_workspace(workspace)

    assert team.list_agents() == ["causal-analyst"]
    assert team.get_agent("causal-analyst").name == "Causal analyst"
