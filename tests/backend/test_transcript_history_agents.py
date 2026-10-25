from konen import TranscriptHistory, TranscriptRound


def _multi_agent_history() -> TranscriptHistory:
    return TranscriptHistory(
        rounds=[
            TranscriptRound(
                agent_uid="bayesian-analyst",
                uid="550e8400-e29b-41d4-a716-446655440000",
                user_prompt="Estimate the posterior probability of the hypothesis.",
                agent_output="The posterior probability is 0.82.",
                metadata={"token_usage": {"input_token_count": 120}},
            ),
            TranscriptRound(
                agent_uid="decision-analyst",
                uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
                user_prompt="Compare the expected utility of the two actions.",
                agent_output=(
                    "Action A has the higher expected utility under the current model."
                ),
                metadata={"token_usage": {"input_token_count": 134}},
            ),
            TranscriptRound(
                agent_uid="bayesian-analyst",
                uid="123e4567-e89b-12d3-a456-426614174000",
                user_prompt="How sensitive is the posterior to the prior scale?",
                agent_output="The posterior is moderately sensitive to wider priors.",
                metadata={"token_usage": {"input_token_count": 148}},
            ),
            TranscriptRound(
                agent_uid="causal-analyst",
                uid="9b2d7c4e-3f18-4a6b-8c21-5d7e9f103456",
                user_prompt="Assess whether the treatment effect is identifiable.",
                agent_output="The effect is identifiable if the assumptions hold.",
                metadata={"token_usage": {"input_token_count": 142}},
            ),
        ]
    )


def test_transcript_history_for_agent() -> None:
    history = _multi_agent_history()

    selected = history.for_agent("bayesian-analyst")

    assert [round.uid for round in selected.rounds] == [
        "550e8400-e29b-41d4-a716-446655440000",
        "123e4567-e89b-12d3-a456-426614174000",
    ]
    selected.rounds[0].metadata["token_usage"]["input_token_count"] = 999
    assert history.rounds[0].metadata["token_usage"]["input_token_count"] == 120


def test_transcript_history_for_agent_empty() -> None:
    history = _multi_agent_history()

    selected = history.for_agent("forecast-analyst")

    assert selected.rounds == []


def test_transcript_history_for_agents() -> None:
    history = _multi_agent_history()

    selected = history.for_agents({"bayesian-analyst", "causal-analyst"})

    assert [round.uid for round in selected.rounds] == [
        "550e8400-e29b-41d4-a716-446655440000",
        "123e4567-e89b-12d3-a456-426614174000",
        "9b2d7c4e-3f18-4a6b-8c21-5d7e9f103456",
    ]
    selected.rounds[2].metadata["token_usage"]["input_token_count"] = 999
    assert history.rounds[3].metadata["token_usage"]["input_token_count"] == 142


def test_transcript_history_by_agent() -> None:
    history = _multi_agent_history()

    grouped = history.by_agent()

    assert list(grouped) == ["bayesian-analyst", "decision-analyst", "causal-analyst"]
    assert [round.uid for round in grouped["bayesian-analyst"].rounds] == [
        "550e8400-e29b-41d4-a716-446655440000",
        "123e4567-e89b-12d3-a456-426614174000",
    ]
    assert [round.uid for round in grouped["decision-analyst"].rounds] == [
        "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    ]
    assert [round.uid for round in grouped["causal-analyst"].rounds] == [
        "9b2d7c4e-3f18-4a6b-8c21-5d7e9f103456"
    ]
    grouped["bayesian-analyst"].rounds[0].metadata["token_usage"][
        "input_token_count"
    ] = 999
    assert history.rounds[0].metadata["token_usage"]["input_token_count"] == 120
