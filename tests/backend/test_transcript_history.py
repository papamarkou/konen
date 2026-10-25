import pytest

from konen import TranscriptHistory, TranscriptRound


def _sample_history() -> TranscriptHistory:
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
                agent_uid="bayesian-analyst",
                uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
                user_prompt="How sensitive is the posterior to the prior scale?",
                agent_output="The posterior is moderately sensitive to wider priors.",
                metadata={"token_usage": {"input_token_count": 148}},
            ),
            TranscriptRound(
                agent_uid="bayesian-analyst",
                uid="123e4567-e89b-12d3-a456-426614174000",
                user_prompt="Summarize the posterior uncertainty.",
                agent_output="Most posterior mass lies between 0.71 and 0.89.",
                metadata={"token_usage": {"input_token_count": 136}},
            ),
        ]
    )


def test_transcript_history_defaults() -> None:
    history = TranscriptHistory()

    assert history.rounds == []
    assert history.first_round is None
    assert history.last_round is None


def test_transcript_history_append() -> None:
    first = TranscriptRound(
        agent_uid="bayesian-analyst",
        uid="550e8400-e29b-41d4-a716-446655440000",
        user_prompt="Estimate the posterior probability of the hypothesis.",
        agent_output="The posterior probability is 0.82.",
    )
    second = TranscriptRound(
        agent_uid="decision-analyst",
        uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        user_prompt="Compare the expected utility of the two actions.",
        agent_output=(
            "Action A has the higher expected utility under the current model."
        ),
    )
    history = TranscriptHistory()

    history.append(first)
    history.append(second)

    assert history.rounds == [first, second]
    assert history.first_round is first
    assert history.last_round is second


def test_transcript_history_contains() -> None:
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        uid="550e8400-e29b-41d4-a716-446655440000",
        user_prompt="Estimate the posterior probability of the hypothesis.",
        agent_output="The posterior probability is 0.82.",
    )
    history = TranscriptHistory(rounds=[round])

    assert history.contains("550e8400-e29b-41d4-a716-446655440000")
    assert not history.contains("123e4567-e89b-12d3-a456-426614174000")


def test_transcript_history_clear() -> None:
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        uid="550e8400-e29b-41d4-a716-446655440000",
        user_prompt="Estimate the posterior probability of the hypothesis.",
        agent_output="The posterior probability is 0.82.",
    )
    history = TranscriptHistory(rounds=[round])

    history.clear()

    assert history.rounds == []
    assert history.first_round is None
    assert history.last_round is None


def test_transcript_history_tail() -> None:
    history = _sample_history()

    selected = history.tail(2)

    assert [round.uid for round in selected.rounds] == [
        "7c9e6679-7425-40de-944b-e07fc1f90ae7",
        "123e4567-e89b-12d3-a456-426614174000",
    ]
    selected.rounds[0].metadata["token_usage"]["input_token_count"] = 999
    assert history.rounds[1].metadata["token_usage"]["input_token_count"] == 148


@pytest.mark.parametrize(
    ("history_limit", "expected_uids"),
    [
        (
            None,
            [
                "550e8400-e29b-41d4-a716-446655440000",
                "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "123e4567-e89b-12d3-a456-426614174000",
            ],
        ),
        (0, []),
        (
            5,
            [
                "550e8400-e29b-41d4-a716-446655440000",
                "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "123e4567-e89b-12d3-a456-426614174000",
            ],
        ),
    ],
)
def test_transcript_history_tail_limits(
    history_limit: int | None, expected_uids: list[str]
) -> None:
    history = _sample_history()

    selected = history.tail(history_limit)

    assert [round.uid for round in selected.rounds] == expected_uids


def test_transcript_history_tail_invalid_limit() -> None:
    history = _sample_history()

    with pytest.raises(ValueError, match="history_limit must be non-negative"):
        history.tail(-1)


def test_transcript_history_after() -> None:
    history = _sample_history()

    selected = history.after("550e8400-e29b-41d4-a716-446655440000")

    assert [round.uid for round in selected.rounds] == [
        "7c9e6679-7425-40de-944b-e07fc1f90ae7",
        "123e4567-e89b-12d3-a456-426614174000",
    ]
    selected.rounds[0].metadata["token_usage"]["input_token_count"] = 999
    assert history.rounds[1].metadata["token_usage"]["input_token_count"] == 148


def test_transcript_history_after_missing() -> None:
    history = _sample_history()

    with pytest.raises(ValueError, match="was not found in history"):
        history.after("9b2d7c4e-3f18-4a6b-8c21-5d7e9f103456")


def test_transcript_history_for_turn() -> None:
    history = _sample_history()

    selected = history.for_turn(2)

    assert selected.uid == "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    selected.metadata["token_usage"]["input_token_count"] = 999
    assert history.rounds[1].metadata["token_usage"]["input_token_count"] == 148


@pytest.mark.parametrize("turn", [0, -1, True, 1.5, "1", 4])
def test_transcript_history_for_turn_invalid(turn: object) -> None:
    history = _sample_history()

    with pytest.raises(ValueError):
        history.for_turn(turn)  # type: ignore[arg-type]


def test_transcript_history_for_turns() -> None:
    history = _sample_history()

    selected = history.for_turns([1, 3])

    assert list(selected) == [1, 3]
    assert selected[1].uid == "550e8400-e29b-41d4-a716-446655440000"
    assert selected[3].uid == "123e4567-e89b-12d3-a456-426614174000"
    selected[3].metadata["token_usage"]["input_token_count"] = 999
    assert history.rounds[2].metadata["token_usage"]["input_token_count"] == 136


def test_transcript_history_for_turns_duplicates() -> None:
    history = _sample_history()

    with pytest.raises(ValueError, match="must not contain duplicates"):
        history.for_turns([1, 1])


def test_transcript_history_for_turns_invalid() -> None:
    history = _sample_history()

    with pytest.raises(ValueError):
        history.for_turns([1, 4])
