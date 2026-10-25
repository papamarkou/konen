import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from konen import TranscriptHistory, TranscriptRound


def _output_history() -> TranscriptHistory:
    return TranscriptHistory(
        rounds=[
            TranscriptRound(
                agent_uid="bayesian-analyst",
                uid="550e8400-e29b-41d4-a716-446655440000",
                user_prompt="Estimate the posterior probability of the hypothesis.",
                agent_output="The posterior probability is 0.82.",
                started_at=datetime(2026, 8, 19, 10, 30, tzinfo=timezone.utc),
                completed_at=datetime(2026, 8, 19, 10, 31, tzinfo=timezone.utc),
                metadata={"engine_uid": "gpt-analysis", "history_limit": 5},
            ),
            TranscriptRound(
                agent_uid="decision-analyst",
                uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
                user_prompt="Compare the expected utility of the two actions.",
                agent_output=(
                    "Action A has the higher expected utility under the current model."
                ),
                started_at=datetime(2026, 8, 19, 10, 32, tzinfo=timezone.utc),
                completed_at=datetime(2026, 8, 19, 10, 33, tzinfo=timezone.utc),
                metadata={"engine_uid": "gpt-analysis", "history_limit": 5},
            ),
        ]
    )


def test_transcript_history_to_markdown() -> None:
    history = _output_history()

    assert history.to_markdown() == (
        "# Transcript history\n\n"
        "## Turn 1\n\n"
        "### Agent\n\n"
        "Agent identifier: bayesian-analyst\n\n"
        "### User prompt\n\n"
        "~~~\n"
        "Estimate the posterior probability of the hypothesis.\n"
        "~~~\n\n"
        "### Agent output\n\n"
        "~~~\n"
        "The posterior probability is 0.82.\n"
        "~~~\n\n"
        "## Turn 2\n\n"
        "### Agent\n\n"
        "Agent identifier: decision-analyst\n\n"
        "### User prompt\n\n"
        "~~~\n"
        "Compare the expected utility of the two actions.\n"
        "~~~\n\n"
        "### Agent output\n\n"
        "~~~\n"
        "Action A has the higher expected utility under the current model.\n"
        "~~~\n"
    )


def test_transcript_history_to_markdown_empty() -> None:
    history = TranscriptHistory()

    assert history.to_markdown() == ""


def test_transcript_history_to_markdown_limit() -> None:
    history = _output_history()

    markdown = history.to_markdown(history_limit=1)

    assert markdown.startswith("# Transcript history\n\n## Turn 2")
    assert "bayesian-analyst" not in markdown
    assert "decision-analyst" in markdown


def test_transcript_history_to_markdown_turns() -> None:
    history = _output_history()

    markdown = history.to_markdown(turns=[2])

    assert markdown.startswith("# Transcript history\n\n## Turn 2")
    assert "bayesian-analyst" not in markdown
    assert "decision-analyst" in markdown


def test_transcript_history_to_markdown_metadata() -> None:
    history = _output_history()

    markdown = history.to_markdown(turns=[1], include_metadata=True)

    assert "### Metadata" in markdown
    assert '"engine_uid": "gpt-analysis"' in markdown
    assert '"history_limit": 5' in markdown


def test_transcript_history_to_markdown_conflict() -> None:
    history = _output_history()

    with pytest.raises(
        ValueError, match="history_limit and turns cannot be used together"
    ):
        history.to_markdown(history_limit=1, turns=[2])


def test_transcript_history_to_dict() -> None:
    history = _output_history()

    assert history.to_dict() == [
        {
            "agent_uid": "bayesian-analyst",
            "uid": "550e8400-e29b-41d4-a716-446655440000",
            "user_prompt": "Estimate the posterior probability of the hypothesis.",
            "agent_output": "The posterior probability is 0.82.",
            "started_at": "2026-08-19T10:30:00+00:00",
            "completed_at": "2026-08-19T10:31:00+00:00",
            "metadata": {"engine_uid": "gpt-analysis", "history_limit": 5},
        },
        {
            "agent_uid": "decision-analyst",
            "uid": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
            "user_prompt": "Compare the expected utility of the two actions.",
            "agent_output": (
                "Action A has the higher expected utility under the current model."
            ),
            "started_at": "2026-08-19T10:32:00+00:00",
            "completed_at": "2026-08-19T10:33:00+00:00",
            "metadata": {"engine_uid": "gpt-analysis", "history_limit": 5},
        },
    ]


def test_transcript_history_save_markdown(tmp_path: Path) -> None:
    history = _output_history()
    path = tmp_path / "exports" / "transcript.md"

    history.save_markdown(path, fence="```")

    assert path.read_text(encoding="utf-8") == history.to_markdown(fence="```")


def test_transcript_history_save_json(tmp_path: Path) -> None:
    history = _output_history()
    path = tmp_path / "exports" / "transcript.json"

    history.save_json(path)

    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == history.to_dict()
    assert text.endswith("\n")
