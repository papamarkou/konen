import pytest

from konen import InteractionState, ProviderRound, TranscriptRound


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


def test_interaction_state_defaults() -> None:
    state = InteractionState()

    assert state.round is None
    assert state.transcript_history.rounds == []
    assert state.provider_history.rounds == []


def test_interaction_state_start_round() -> None:
    state = InteractionState()

    state.start_round(
        agent_uid="bayesian-analyst",
        team_context=_team_context(),
        user_prompt=(
            "Assess whether the posterior probability remains above 0.75 "
            "under the wider prior."
        ),
        memory_context={
            "research_notes.md": "The baseline posterior probability is 0.82."
        },
        metadata={"engine_uid": "gpt-analysis"},
    )

    assert state.round is not None
    assert state.round.agent_uid == "bayesian-analyst"
    assert state.round.transcript.user_prompt == (
        "Assess whether the posterior probability remains above 0.75 "
        "under the wider prior."
    )
    assert state.round.transcript.started_at is not None
    assert state.round.memory_context == {
        "research_notes.md": "The baseline posterior probability is 0.82."
    }
    assert state.round.transcript.metadata == {"engine_uid": "gpt-analysis"}


def test_interaction_state_cannot_start_incomplete_round() -> None:
    state = InteractionState()
    state.start_round(
        agent_uid="bayesian-analyst",
        team_context=_team_context(),
        user_prompt="Estimate the posterior probability under the revised prior.",
    )

    with pytest.raises(RuntimeError, match="current round is incomplete"):
        state.start_round(
            agent_uid="decision-analyst",
            team_context=_team_context(),
            user_prompt="Compare the expected utility of the available actions.",
        )


def test_interaction_state_append_to_history() -> None:
    state = InteractionState()
    state.start_round(
        agent_uid="bayesian-analyst",
        team_context=_team_context(),
        user_prompt="Estimate the posterior probability under the revised prior.",
        metadata={"token_usage": {"input_token_count": 128}},
    )
    assert state.round is not None
    state.round.transcript.agent_output = "The revised posterior probability is 0.79."
    state.round.transcript.complete()

    state.append_to_history()

    assert len(state.transcript_history.rounds) == 1
    stored = state.transcript_history.rounds[0]
    assert stored.uid == state.round.transcript.uid
    assert stored.agent_output == "The revised posterior probability is 0.79."

    state.round.transcript.metadata["token_usage"]["input_token_count"] = 999
    assert stored.metadata["token_usage"]["input_token_count"] == 128


def test_interaction_state_append_without_round() -> None:
    state = InteractionState()

    with pytest.raises(ValueError, match="Start a round before appending"):
        state.append_to_history()


def test_interaction_state_append_incomplete_round() -> None:
    state = InteractionState()
    state.start_round(
        agent_uid="bayesian-analyst",
        team_context=_team_context(),
        user_prompt="Estimate the posterior probability under the revised prior.",
    )

    with pytest.raises(RuntimeError, match="before it is completed"):
        state.append_to_history()


def test_interaction_state_append_provider_round() -> None:
    transcript = TranscriptRound(
        agent_uid="bayesian-analyst",
        uid="550e8400-e29b-41d4-a716-446655440000",
        user_prompt="Estimate the posterior probability under the revised prior.",
        agent_output="The revised posterior probability is 0.79.",
    )
    provider_round = ProviderRound(
        transcript_uid="550e8400-e29b-41d4-a716-446655440000",
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
        state={"response": {"id": "resp_01HXYZ123", "status": "completed"}},
    )
    state = InteractionState()
    state.transcript_history.append(transcript)

    state.append_provider_round(provider_round)

    assert len(state.provider_history.rounds) == 1
    stored = state.provider_history.rounds[0]
    assert stored.transcript_uid == transcript.uid

    provider_round.state["response"]["status"] = "changed"
    assert stored.state["response"]["status"] == "completed"


def test_interaction_state_provider_requires_transcript() -> None:
    state = InteractionState()
    provider_round = ProviderRound(
        transcript_uid="550e8400-e29b-41d4-a716-446655440000",
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
        state={"response_id": "resp_01HXYZ123"},
    )

    with pytest.raises(ValueError, match="because it is not in history"):
        state.append_provider_round(provider_round)


def test_interaction_state_clear_round() -> None:
    state = InteractionState()
    state.start_round(
        agent_uid="bayesian-analyst",
        team_context=_team_context(),
        user_prompt="Estimate the posterior probability under the revised prior.",
    )

    state.clear_round()

    assert state.round is None


def test_interaction_state_clear_provider_history() -> None:
    state = InteractionState()
    state.provider_history.rounds.append(
        ProviderRound(
            transcript_uid="550e8400-e29b-41d4-a716-446655440000",
            agent_uid="bayesian-analyst",
            engine_uid="gpt-analysis",
            model_family="gpt",
            model_name="gpt-5.6-luna",
            history_start_uid="550e8400-e29b-41d4-a716-446655440000",
            state={"response_id": "resp_01HXYZ123"},
        )
    )

    state.clear_provider_history()

    assert state.provider_history.rounds == []


def test_interaction_state_clear_history() -> None:
    state = InteractionState()
    state.transcript_history.append(
        TranscriptRound(
            agent_uid="bayesian-analyst",
            uid="550e8400-e29b-41d4-a716-446655440000",
            user_prompt="Estimate the posterior probability under the revised prior.",
            agent_output="The revised posterior probability is 0.79.",
        )
    )
    state.provider_history.rounds.append(
        ProviderRound(
            transcript_uid="550e8400-e29b-41d4-a716-446655440000",
            agent_uid="bayesian-analyst",
            engine_uid="gpt-analysis",
            model_family="gpt",
            model_name="gpt-5.6-luna",
            history_start_uid="550e8400-e29b-41d4-a716-446655440000",
            state={"response_id": "resp_01HXYZ123"},
        )
    )

    state.clear_history()

    assert state.transcript_history.rounds == []
    assert state.provider_history.rounds == []


def test_interaction_state_clear() -> None:
    state = InteractionState()
    state.start_round(
        agent_uid="bayesian-analyst",
        team_context=_team_context(),
        user_prompt="Estimate the posterior probability under the revised prior.",
    )
    state.transcript_history.append(
        TranscriptRound(
            agent_uid="decision-analyst",
            uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
            user_prompt="Compare the expected utility of the available actions.",
            agent_output="Action A has the higher expected utility.",
        )
    )
    state.provider_history.rounds.append(
        ProviderRound(
            transcript_uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
            agent_uid="decision-analyst",
            engine_uid="gpt-analysis",
            model_family="gpt",
            model_name="gpt-5.6-luna",
            history_start_uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
            state={"response_id": "resp_01HXYZ456"},
        )
    )

    state.clear()

    assert state.round is None
    assert state.transcript_history.rounds == []
    assert state.provider_history.rounds == []
