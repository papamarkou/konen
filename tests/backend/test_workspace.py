import json
from pathlib import Path

import pytest

from konen import Workspace


def test_workspace_paths(tmp_path: Path) -> None:
    root = tmp_path / "bayesian-sensitivity-study"
    workspace = Workspace(root_path=root)

    assert workspace.root_path == root.resolve()
    assert workspace.memory_path == root.resolve() / "memory"
    assert workspace.scratchpad_path == root.resolve() / "scratchpad"
    assert workspace.config_path == root.resolve() / "config"
    assert workspace.agents_path == root.resolve() / "config" / "agents.json"
    assert workspace.engines_path == root.resolve() / "config" / "engines.json"
    assert workspace.agent_engine_map_path == (
        root.resolve() / "config" / "agent_engine_map.json"
    )
    assert not workspace.is_initialized


def test_workspace_custom_paths(tmp_path: Path) -> None:
    root = tmp_path / "decision-analysis-project"
    workspace = Workspace(
        root_path=root,
        memory_dir_name="research_memory",
        scratchpad_dir_name="agent_notes",
        config_dir_name="settings",
        agents_filename="team.json",
        engines_filename="models.json",
        agent_engine_map_filename="routing.json",
    )

    assert workspace.memory_path == root.resolve() / "research_memory"
    assert workspace.scratchpad_path == root.resolve() / "agent_notes"
    assert workspace.config_path == root.resolve() / "settings"
    assert workspace.agents_path == root.resolve() / "settings" / "team.json"
    assert workspace.engines_path == root.resolve() / "settings" / "models.json"
    assert workspace.agent_engine_map_path == (
        root.resolve() / "settings" / "routing.json"
    )


def test_workspace_initialize(tmp_path: Path) -> None:
    workspace = Workspace(root_path=tmp_path / "bayesian-sensitivity-study")

    workspace.initialize()

    assert workspace.is_initialized
    assert workspace.root_path.is_dir()
    assert workspace.memory_path.is_dir()
    assert workspace.scratchpad_path.is_dir()
    assert workspace.config_path.is_dir()
    assert workspace.agents_path.is_file()
    assert workspace.engines_path.is_file()
    assert workspace.agent_engine_map_path.is_file()
    assert json.loads(workspace.agents_path.read_text(encoding="utf-8")) == {
        "version": "1.0",
        "agents": {},
    }
    assert json.loads(workspace.engines_path.read_text(encoding="utf-8")) == {
        "version": "1.0",
        "engines": {},
    }
    assert json.loads(workspace.agent_engine_map_path.read_text(encoding="utf-8")) == {
        "version": "1.0",
        "agent_engine_map": {},
    }


def test_workspace_initialize_idempotent(tmp_path: Path) -> None:
    workspace = Workspace(root_path=tmp_path / "bayesian-sensitivity-study")
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
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    workspace.initialize()

    assert json.loads(workspace.agents_path.read_text(encoding="utf-8")) == {
        "version": "1.0",
        "agents": {
            "bayesian-analyst": {
                "name": "Bayesian analyst",
                "description": (
                    "Evaluates uncertainty using Bayesian inference and "
                    "posterior reasoning."
                ),
            }
        },
    }


def test_workspace_ensure_initialized(tmp_path: Path) -> None:
    workspace = Workspace(root_path=tmp_path / "bayesian-sensitivity-study")

    with pytest.raises(RuntimeError, match="Workspace is not initialized"):
        workspace.ensure_initialized()

    workspace.initialize()
    workspace.ensure_initialized()


@pytest.mark.parametrize("file_type", ["memory", "scratchpad"])
@pytest.mark.parametrize(
    ("initial", "separator", "expected"),
    [
        ("", None, "New note\n"),
        ("Existing\n", None, "Existing\n\nNew note\n"),
        ("Existing", None, "Existing\n\nNew note\n"),
        ("Existing\n\n", None, "Existing\n\nNew note\n"),
        ("Existing\n", "", "Existing\nNew note\n"),
        ("Existing\n", "---\n", "Existing\n---\nNew note\n"),
    ],
)
def test_workspace_append_text(
    tmp_path: Path,
    file_type: str,
    initial: str,
    separator: str | None,
    expected: str,
) -> None:
    workspace = Workspace(root_path=tmp_path / "bayesian-sensitivity-study")
    workspace.initialize()
    kwargs = {} if separator is None else {"separator": separator}

    if file_type == "memory":
        path = workspace.memory_path / "notes.md"
        path.write_text(initial, encoding="utf-8")
        workspace.append_to_memory("notes.md", "New note\n", **kwargs)
    else:
        workspace.add_agent("analyst", "Analyst", "Analyzes the problem.")
        path = workspace.scratchpad_path / "analyst.md"
        path.write_text(initial, encoding="utf-8")
        workspace.append_to_scratchpad("analyst", "New note\n", **kwargs)

    assert path.read_text(encoding="utf-8") == expected
