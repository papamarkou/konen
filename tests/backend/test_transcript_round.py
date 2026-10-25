from datetime import datetime, timezone
from uuid import UUID

import pytest

from konen import TranscriptRound


def test_transcript_round_preserves_fields() -> None:
    started_at = datetime(2026, 8, 19, 10, 30, tzinfo=timezone.utc)
    completed_at = datetime(2026, 8, 19, 10, 31, tzinfo=timezone.utc)
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        user_prompt="Estimate the posterior probability of the hypothesis.",
        agent_output="The posterior probability is 0.82.",
        started_at=started_at,
        completed_at=completed_at,
        metadata={"history_limit": 5},
    )

    assert round.agent_uid == "bayesian-analyst"
    assert round.uid == "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    assert round.user_prompt == "Estimate the posterior probability of the hypothesis."
    assert round.agent_output == "The posterior probability is 0.82."
    assert round.started_at == started_at
    assert round.completed_at == completed_at
    assert round.metadata == {"history_limit": 5}


def test_transcript_round_defaults() -> None:
    round = TranscriptRound(agent_uid="bayesian-analyst")

    assert UUID(round.uid).version == 4
    assert round.user_prompt is None
    assert round.agent_output is None
    assert round.started_at is None
    assert round.completed_at is None
    assert round.metadata == {}


def test_transcript_round_deep_copies_metadata_input() -> None:
    metadata = {"token_usage": {"input_token_count": 120}}
    round = TranscriptRound(agent_uid="bayesian-analyst", metadata=metadata)

    metadata["token_usage"]["input_token_count"] = 999

    assert round.metadata == {"token_usage": {"input_token_count": 120}}


def test_transcript_round_start_utc_timestamp() -> None:
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        user_prompt="Estimate the posterior probability of the hypothesis.",
    )
    before = datetime.now(timezone.utc)

    round.start()

    after = datetime.now(timezone.utc)
    assert round.started_at is not None
    assert before <= round.started_at <= after
    assert round.started_at.tzinfo == timezone.utc
    assert round.completed_at is None


def test_transcript_round_cannot_start_twice() -> None:
    round = TranscriptRound(agent_uid="bayesian-analyst")
    round.start()

    with pytest.raises(RuntimeError, match="already started"):
        round.start()


def test_transcript_round_cannot_complete_before_start() -> None:
    round = TranscriptRound(agent_uid="bayesian-analyst")

    with pytest.raises(RuntimeError, match="cannot complete before it starts"):
        round.complete()


def test_transcript_round_complete_utc_timestamp() -> None:
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        user_prompt="Estimate the posterior probability of the hypothesis.",
    )
    round.start()
    before = datetime.now(timezone.utc)

    round.complete()

    after = datetime.now(timezone.utc)
    assert round.completed_at is not None
    assert before <= round.completed_at <= after
    assert round.completed_at.tzinfo == timezone.utc
    assert round.started_at <= round.completed_at


def test_transcript_round_cannot_complete_twice() -> None:
    round = TranscriptRound(agent_uid="bayesian-analyst")
    round.start()
    round.complete()

    with pytest.raises(RuntimeError, match="already completed"):
        round.complete()


def test_transcript_round_update_metadata() -> None:
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        metadata={"history_limit": 5, "engine_uid": "gpt-analysis"},
    )

    round.update_metadata(
        {"history_limit": 3, "provider_response_id": "resp_01HXYZ123"}
    )

    assert round.metadata == {
        "history_limit": 3,
        "engine_uid": "gpt-analysis",
        "provider_response_id": "resp_01HXYZ123",
    }


def test_transcript_round_update_metadata_deep_copy() -> None:
    metadata = {"token_usage": {"output_token_count": 48}}
    round = TranscriptRound(agent_uid="bayesian-analyst")

    round.update_metadata(metadata)
    metadata["token_usage"]["output_token_count"] = 999

    assert round.metadata == {"token_usage": {"output_token_count": 48}}


def test_transcript_round_to_dict() -> None:
    started_at = datetime(2026, 8, 19, 10, 30, tzinfo=timezone.utc)
    completed_at = datetime(2026, 8, 19, 10, 31, tzinfo=timezone.utc)
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        user_prompt="Estimate the posterior probability of the hypothesis.",
        agent_output="The posterior probability is 0.82.",
        started_at=started_at,
        completed_at=completed_at,
        metadata={
            "engine_uid": "gpt-analysis",
            "provider_response_id": "resp_01HXYZ123",
        },
    )

    assert round.to_dict() == {
        "agent_uid": "bayesian-analyst",
        "uid": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
        "user_prompt": "Estimate the posterior probability of the hypothesis.",
        "agent_output": "The posterior probability is 0.82.",
        "started_at": "2026-08-19T10:30:00+00:00",
        "completed_at": "2026-08-19T10:31:00+00:00",
        "metadata": {
            "engine_uid": "gpt-analysis",
            "provider_response_id": "resp_01HXYZ123",
        },
    }


def test_transcript_round_to_dict_deep_copy() -> None:
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        metadata={"token_usage": {"input_token_count": 120}},
    )

    data = round.to_dict()
    data["metadata"]["token_usage"]["input_token_count"] = 999

    assert round.metadata == {"token_usage": {"input_token_count": 120}}


def test_transcript_round_to_markdown() -> None:
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        user_prompt="Compare the posterior odds for the two hypotheses.",
        agent_output=("The posterior odds favor the first hypothesis by about 4:1."),
    )

    assert round.to_markdown() == (
        "# Transcript round\n\n"
        "### Agent\n\n"
        "Agent identifier: bayesian-analyst\n\n"
        "### User prompt\n\n"
        "~~~\n"
        "Compare the posterior odds for the two hypotheses.\n"
        "~~~\n\n"
        "### Agent output\n\n"
        "~~~\n"
        "The posterior odds favor the first hypothesis by about 4:1.\n"
        "~~~\n"
    )


def test_transcript_round_to_markdown_custom_fence() -> None:
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        user_prompt="Summarize the posterior uncertainty.",
        agent_output="Most posterior mass lies between 0.71 and 0.89.",
    )

    assert round.to_markdown(fence="```") == (
        "# Transcript round\n\n"
        "### Agent\n\n"
        "Agent identifier: bayesian-analyst\n\n"
        "### User prompt\n\n"
        "```\n"
        "Summarize the posterior uncertainty.\n"
        "```\n\n"
        "### Agent output\n\n"
        "```\n"
        "Most posterior mass lies between 0.71 and 0.89.\n"
        "```\n"
    )


def test_transcript_round_to_markdown_empty_text() -> None:
    round = TranscriptRound(agent_uid="bayesian-analyst")

    assert round.to_markdown() == (
        "# Transcript round\n\n"
        "### Agent\n\n"
        "Agent identifier: bayesian-analyst\n\n"
        "### User prompt\n\n"
        "~~~\n\n~~~\n\n"
        "### Agent output\n\n"
        "~~~\n\n~~~\n"
    )


def test_transcript_round_to_markdown_turn() -> None:
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        user_prompt="How sensitive is the posterior to the prior scale?",
        agent_output=(
            "The posterior is moderately sensitive when the prior scale exceeds 2."
        ),
    )

    assert round.to_markdown(turn=3) == (
        "## Turn 3\n\n"
        "### Agent\n\n"
        "Agent identifier: bayesian-analyst\n\n"
        "### User prompt\n\n"
        "~~~\n"
        "How sensitive is the posterior to the prior scale?\n"
        "~~~\n\n"
        "### Agent output\n\n"
        "~~~\n"
        "The posterior is moderately sensitive when the prior scale exceeds 2.\n"
        "~~~\n"
    )


def test_transcript_round_to_markdown_metadata() -> None:
    round = TranscriptRound(
        agent_uid="bayesian-analyst",
        user_prompt="Report the posterior mean and uncertainty.",
        agent_output="The posterior mean is 0.81 with a 95% interval of [0.72, 0.89].",
        metadata={
            "engine_uid": "gpt-analysis",
            "provider_response_id": "resp_01HXYZ123",
        },
    )

    assert round.to_markdown(include_metadata=True) == (
        "# Transcript round\n\n"
        "### Agent\n\n"
        "Agent identifier: bayesian-analyst\n\n"
        "### Metadata\n\n"
        "~~~\n"
        "{\n"
        '  "engine_uid": "gpt-analysis",\n'
        '  "provider_response_id": "resp_01HXYZ123"\n'
        "}\n"
        "~~~\n\n"
        "### User prompt\n\n"
        "~~~\n"
        "Report the posterior mean and uncertainty.\n"
        "~~~\n\n"
        "### Agent output\n\n"
        "~~~\n"
        "The posterior mean is 0.81 with a 95% interval of [0.72, 0.89].\n"
        "~~~\n"
    )


@pytest.mark.parametrize("turn", [0, -1, True, 1.5, "1"])
def test_transcript_round_to_markdown_invalid_turn(turn: object) -> None:
    round = TranscriptRound(agent_uid="bayesian-analyst")

    with pytest.raises(ValueError, match="turn must be a positive integer or None"):
        round.to_markdown(turn=turn)  # type: ignore[arg-type]
