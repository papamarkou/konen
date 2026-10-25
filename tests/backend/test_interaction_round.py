from datetime import datetime, timezone

import pytest

from konen import InteractionRound


def _team_context() -> dict[str, dict[str, str]]:
    return {
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


def test_interaction_round_preserves_fields() -> None:
    interaction = InteractionRound(
        agent_uid="bayesian-analyst",
        team_context=_team_context(),
        system_context="Analyze uncertainty and report assumptions clearly.",
        memory_context={
            "research_notes.md": "The prior odds favor the first hypothesis by 2:1."
        },
        scratchpad_context={
            "working_assumptions.md": "Assume the observations are conditionally independent."
        },
    )

    assert interaction.agent_uid == "bayesian-analyst"
    assert interaction.system_context == (
        "Analyze uncertainty and report assumptions clearly."
    )
    assert interaction.memory_context == {
        "research_notes.md": "The prior odds favor the first hypothesis by 2:1."
    }
    assert interaction.scratchpad_context == {
        "working_assumptions.md": "Assume the observations are conditionally independent."
    }
    assert interaction.transcript.agent_uid == "bayesian-analyst"
    assert interaction.transcript.user_prompt is None
    assert interaction.transcript.started_at is None
    assert interaction.transcript.completed_at is None
    assert interaction.transcript.metadata == {}


def test_interaction_round_copies_inputs() -> None:
    team_context = _team_context()
    memory_context = {
        "research_notes.md": "The prior odds favor the first hypothesis by 2:1."
    }
    scratchpad_context = {
        "working_assumptions.md": "Assume the observations are conditionally independent."
    }
    interaction = InteractionRound(
        agent_uid="bayesian-analyst",
        team_context=team_context,
        memory_context=memory_context,
        scratchpad_context=scratchpad_context,
    )

    team_context["bayesian-analyst"]["description"] = "Changed"
    memory_context["research_notes.md"] = "Changed"
    scratchpad_context["working_assumptions.md"] = "Changed"

    assert interaction.team_context["bayesian-analyst"]["description"] == (
        "Evaluates uncertainty using Bayesian inference and posterior reasoning."
    )
    assert interaction.memory_context["research_notes.md"] == (
        "The prior odds favor the first hypothesis by 2:1."
    )
    assert interaction.scratchpad_context["working_assumptions.md"] == (
        "Assume the observations are conditionally independent."
    )


def test_interaction_round_identity_context() -> None:
    interaction = InteractionRound(
        agent_uid="bayesian-analyst", team_context=_team_context()
    )

    assert interaction.identity_context == {
        "name": "Bayesian analyst",
        "description": (
            "Evaluates uncertainty using Bayesian inference and posterior reasoning."
        ),
    }


def test_interaction_round_missing_agent() -> None:
    with pytest.raises(ValueError, match="was not found in team context"):
        InteractionRound(agent_uid="causal-analyst", team_context=_team_context())


def test_interaction_round_invalid_team_context() -> None:
    team_context = _team_context()
    del team_context["decision-analyst"]["description"]

    with pytest.raises(ValueError, match="must contain 'name' and 'description'"):
        InteractionRound(agent_uid="bayesian-analyst", team_context=team_context)


def test_interaction_round_from_inputs() -> None:
    metadata = {"engine_uid": "gpt-analysis", "token_usage": {"input_token_count": 128}}
    before = datetime.now(timezone.utc)

    interaction = InteractionRound.from_inputs(
        agent_uid="bayesian-analyst",
        team_context=_team_context(),
        user_prompt=(
            "Assess whether the posterior probability remains above 0.75 "
            "under the wider prior."
        ),
        memory_context={
            "research_notes.md": "The baseline posterior probability is 0.82."
        },
        scratchpad_context={"sensitivity_plan.md": "Compare prior scales of 1 and 2."},
        metadata=metadata,
    )

    after = datetime.now(timezone.utc)
    metadata["token_usage"]["input_token_count"] = 999

    assert interaction.transcript.user_prompt == (
        "Assess whether the posterior probability remains above 0.75 "
        "under the wider prior."
    )
    assert interaction.transcript.started_at is not None
    assert before <= interaction.transcript.started_at <= after
    assert interaction.transcript.started_at.tzinfo == timezone.utc
    assert interaction.transcript.completed_at is None
    assert interaction.transcript.metadata == {
        "engine_uid": "gpt-analysis",
        "token_usage": {"input_token_count": 128},
    }


def test_interaction_round_from_inputs_no_prompt() -> None:
    interaction = InteractionRound.from_inputs(
        agent_uid="bayesian-analyst",
        team_context=_team_context(),
        metadata={"engine_uid": "gpt-analysis"},
    )

    assert interaction.transcript.user_prompt is None
    assert interaction.transcript.started_at is None
    assert interaction.transcript.metadata == {"engine_uid": "gpt-analysis"}


def test_interaction_round_to_dict() -> None:
    interaction = InteractionRound(
        agent_uid="bayesian-analyst",
        team_context=_team_context(),
        system_context="Analyze uncertainty and report assumptions clearly.",
        memory_context={
            "research_notes.md": "The prior odds favor the first hypothesis by 2:1."
        },
        scratchpad_context={
            "working_assumptions.md": "Assume the observations are conditionally independent."
        },
    )
    interaction.transcript.user_prompt = (
        "Estimate the posterior probability under the revised prior."
    )
    interaction.transcript.update_metadata({"engine_uid": "gpt-analysis"})

    assert interaction.to_dict() == {
        "agent_uid": "bayesian-analyst",
        "team_context": _team_context(),
        "system_context": "Analyze uncertainty and report assumptions clearly.",
        "memory_context": {
            "research_notes.md": "The prior odds favor the first hypothesis by 2:1."
        },
        "scratchpad_context": {
            "working_assumptions.md": "Assume the observations are conditionally independent."
        },
        "transcript": {
            "agent_uid": "bayesian-analyst",
            "uid": interaction.transcript.uid,
            "user_prompt": "Estimate the posterior probability under the revised prior.",
            "agent_output": None,
            "started_at": None,
            "completed_at": None,
            "metadata": {"engine_uid": "gpt-analysis"},
        },
    }


def test_interaction_round_to_dict_copy() -> None:
    interaction = InteractionRound(
        agent_uid="bayesian-analyst",
        team_context=_team_context(),
        memory_context={
            "research_notes.md": "The prior odds favor the first hypothesis by 2:1."
        },
    )
    interaction.transcript.update_metadata({"token_usage": {"input_token_count": 128}})

    data = interaction.to_dict()
    data["team_context"]["bayesian-analyst"]["name"] = "Changed"
    data["memory_context"]["research_notes.md"] = "Changed"
    data["transcript"]["metadata"]["token_usage"]["input_token_count"] = 999

    assert interaction.team_context["bayesian-analyst"]["name"] == "Bayesian analyst"
    assert interaction.memory_context["research_notes.md"] == (
        "The prior odds favor the first hypothesis by 2:1."
    )
    assert interaction.transcript.metadata["token_usage"]["input_token_count"] == 128
