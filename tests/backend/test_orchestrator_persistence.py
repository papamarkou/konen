from pathlib import Path

import pytest

from konen import InteractionState, Orchestrator, TranscriptRound, Workspace


def _workspace(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    workspace.add_agent(
        agent_uid="analyst",
        name="Analyst",
        description="Analyzes the current problem.",
    )
    workspace.add_agent(
        agent_uid="reviewer",
        name="Reviewer",
        description="Reviews the analysis.",
    )
    return workspace


def _completed_transcript() -> TranscriptRound:
    transcript = TranscriptRound(
        agent_uid="analyst",
        user_prompt="Assess the evidence.",
        agent_output="The evidence supports the hypothesis.",
    )
    transcript.start()
    transcript.complete()
    return transcript


def _orchestrator(tmp_path: Path) -> Orchestrator:
    state = InteractionState()
    state.transcript_history.append(_completed_transcript())
    return Orchestrator.from_workspace(
        workspace=_workspace(tmp_path),
        interaction_state=state,
    )


def test_orchestrator_append_last_turn_to_memory(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)

    orchestrator.append_last_turn_to_memory("findings.md")

    text = orchestrator.workspace.read_workspace_file("memory/findings.md")
    assert "# Transcript round" in text
    assert "Agent identifier: analyst" in text
    assert "Assess the evidence." in text
    assert "The evidence supports the hypothesis." in text


def test_orchestrator_append_last_turn_to_scratchpad(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)

    orchestrator.append_last_turn_to_scratchpad("reviewer")

    text = orchestrator.workspace.read_workspace_file("scratchpad/reviewer.md")
    assert "# Transcript round" in text
    assert "Agent identifier: analyst" in text
    assert "Assess the evidence." in text
    assert "The evidence supports the hypothesis." in text


def test_orchestrator_append_last_turn_requires_completed_turn(tmp_path: Path) -> None:
    orchestrator = Orchestrator.from_workspace(_workspace(tmp_path))

    with pytest.raises(ValueError, match="No completed transcript turn is available"):
        orchestrator.append_last_turn_to_memory("findings.md")

    with pytest.raises(ValueError, match="No completed transcript turn is available"):
        orchestrator.append_last_turn_to_scratchpad("analyst")


def test_orchestrator_append_last_turn_requires_configured_scratchpad_agent(
    tmp_path: Path,
) -> None:
    orchestrator = _orchestrator(tmp_path)

    with pytest.raises(ValueError, match="missing"):
        orchestrator.append_last_turn_to_scratchpad("missing")
