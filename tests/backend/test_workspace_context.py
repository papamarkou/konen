import json
from pathlib import Path

import pytest

from konen import Workspace


def _write_agents_config(workspace: Workspace) -> None:
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


def test_workspace_list_files(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    (workspace.memory_path / "prior_notes.md").write_text(
        "The baseline prior odds favor the first hypothesis by 2:1.\n", encoding="utf-8"
    )
    literature_path = workspace.memory_path / "literature" / "robustness.md"
    literature_path.parent.mkdir(parents=True)
    literature_path.write_text(
        "Posterior conclusions remain stable across moderate prior scales.\n",
        encoding="utf-8",
    )
    (workspace.scratchpad_path / "bayesian-analyst.md").write_text(
        "Compare prior scales of 1 and 2.\n", encoding="utf-8"
    )

    files = workspace.list_workspace_files()

    assert files == {
        "memory": ["literature/robustness.md", "prior_notes.md"],
        "scratchpad": ["bayesian-analyst.md"],
        "config": ["agent_engine_map.json", "agents.json", "engines.json"],
    }


def test_workspace_read_file(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    path = workspace.memory_path / "prior_notes.md"
    path.write_text("The baseline posterior probability is 0.82.\n", encoding="utf-8")

    assert workspace.read_workspace_file("memory/prior_notes.md") == (
        "The baseline posterior probability is 0.82.\n"
    )


def test_workspace_read_file_missing(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()

    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        workspace.read_workspace_file("memory/posterior_summary.md")


@pytest.mark.parametrize("filename", ["", "../outside.md", "project_notes.md"])
def test_workspace_read_file_invalid(tmp_path: Path, filename: str) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()

    with pytest.raises(ValueError):
        workspace.read_workspace_file(filename)


def test_workspace_load_memory_context(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    (workspace.memory_path / "prior_notes.md").write_text(
        "The baseline prior odds favor the first hypothesis by 2:1.\n", encoding="utf-8"
    )
    sensitivity_path = workspace.memory_path / "analysis" / "sensitivity.md"
    sensitivity_path.parent.mkdir(parents=True)
    sensitivity_path.write_text(
        "The posterior remains above 0.75 under the wider prior.\n", encoding="utf-8"
    )

    assert workspace.load_memory_context() == {
        "memory/analysis/sensitivity.md": (
            "The posterior remains above 0.75 under the wider prior.\n"
        ),
        "memory/prior_notes.md": (
            "The baseline prior odds favor the first hypothesis by 2:1.\n"
        ),
    }


def test_workspace_load_selected_memory_context(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    (workspace.memory_path / "prior_notes.md").write_text(
        "The baseline prior probability is 0.40.\n", encoding="utf-8"
    )
    (workspace.memory_path / "posterior_summary.md").write_text(
        "The posterior probability is 0.82.\n", encoding="utf-8"
    )

    assert workspace.load_memory_context(("prior_notes.md",)) == {
        "memory/prior_notes.md": "The baseline prior probability is 0.40.\n"
    }
    assert workspace.load_memory_context(()) == {}


def test_workspace_load_memory_context_includes_new_files(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    (workspace.memory_path / "prior_notes.md").write_text(
        "The baseline prior probability is 0.40.\n", encoding="utf-8"
    )

    assert workspace.load_memory_context() == {
        "memory/prior_notes.md": "The baseline prior probability is 0.40.\n"
    }

    (workspace.memory_path / "posterior_summary.md").write_text(
        "The posterior probability is 0.82.\n", encoding="utf-8"
    )

    assert workspace.load_memory_context() == {
        "memory/posterior_summary.md": "The posterior probability is 0.82.\n",
        "memory/prior_notes.md": "The baseline prior probability is 0.40.\n",
    }
    assert workspace.load_memory_context(("prior_notes.md",)) == {
        "memory/prior_notes.md": "The baseline prior probability is 0.40.\n"
    }


def test_workspace_load_selected_memory_missing_file(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()

    with pytest.raises(FileNotFoundError, match="Memory file does not exist"):
        workspace.load_memory_context(("prior_notes.md",))


def test_workspace_load_selected_memory_invalid_path(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()

    with pytest.raises(ValueError, match="path escapes memory directory"):
        workspace.load_memory_context(("../outside.md",))


def test_workspace_load_scratchpad_context(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    _write_agents_config(workspace)
    (workspace.scratchpad_path / "bayesian-analyst.md").write_text(
        "Compare prior scales of 1 and 2 before reporting the conclusion.\n",
        encoding="utf-8",
    )

    assert workspace.load_scratchpad_context("bayesian-analyst") == {
        "scratchpad/bayesian-analyst.md": (
            "Compare prior scales of 1 and 2 before reporting the conclusion.\n"
        )
    }


def test_workspace_load_scratchpad_invalid_agent(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()

    with pytest.raises(ValueError, match="Invalid agent UID"):
        workspace.load_scratchpad_context("Bayesian analyst")


def test_workspace_load_scratchpad_missing_agent(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    _write_agents_config(workspace)

    with pytest.raises(ValueError, match="Agent 'causal-analyst' does not exist"):
        workspace.load_scratchpad_context("causal-analyst")


def test_workspace_load_scratchpad_missing_file(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    _write_agents_config(workspace)

    with pytest.raises(FileNotFoundError, match="Scratchpad does not exist"):
        workspace.load_scratchpad_context("bayesian-analyst")


def test_workspace_load_configs(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    _write_agents_config(workspace)
    workspace.engines_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "engines": {
                    "gpt-analysis": {
                        "model_family": "gpt",
                        "model_name": "gpt-5.6-luna",
                        "config": {"reasoning_effort": "high"},
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    workspace.agent_engine_map_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "agent_engine_map": {"bayesian-analyst": "gpt-analysis"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    assert workspace.load_agents_config()["agents"]["bayesian-analyst"]["name"] == (
        "Bayesian analyst"
    )
    assert workspace.load_engines_config()["engines"]["gpt-analysis"] == {
        "model_family": "gpt",
        "model_name": "gpt-5.6-luna",
        "config": {"reasoning_effort": "high"},
    }
    assert workspace.load_agent_engine_map_config()["agent_engine_map"] == {
        "bayesian-analyst": "gpt-analysis"
    }


def test_workspace_invalid_json(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    workspace.agents_path.write_text('{"version": "1.0", "agents": ', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in agents config"):
        workspace.load_agents_config()


def test_workspace_invalid_schema(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    workspace.engines_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "engines": {
                    "gpt-analysis": {
                        "model_family": "openai",
                        "model_name": "gpt-5.6-luna",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid engines schema"):
        workspace.load_engines_config()


def test_workspace_invalid_mapping(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    _write_agents_config(workspace)
    workspace.agent_engine_map_path.write_text(
        json.dumps(
            {"version": "1.0", "agent_engine_map": {"bayesian-analyst": "gpt-analysis"}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="references unknown engine 'gpt-analysis'"):
        workspace.load_agent_engine_map_config()
