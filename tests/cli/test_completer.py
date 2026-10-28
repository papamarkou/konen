from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from prompt_toolkit.completion import CompleteEvent, Completion
from prompt_toolkit.document import Document

from konen import AIAgent, AITeam, InteractionState, TranscriptRound
from konen.cli.completer import CLICompleter


@dataclass
class _Workspace:
    files: dict[str, list[str]]

    def list_workspace_files(self) -> dict[str, list[str]]:
        return {category: list(filenames) for category, filenames in self.files.items()}


@dataclass
class _Pool:
    engine_uids: list[str]

    def list_engines(self) -> list[str]:
        return sorted(self.engine_uids)


def _completer() -> CLICompleter:
    team = AITeam(
        [
            AIAgent(
                name="Bayesian analyst",
                description="Evaluates uncertainty using Bayesian inference.",
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
    )
    state = InteractionState()
    state.transcript_history.append(
        TranscriptRound(
            agent_uid="bayesian-analyst",
            user_prompt="Estimate the posterior probability under the baseline prior.",
            agent_output="The baseline posterior probability is 0.82.",
        )
    )
    state.transcript_history.append(
        TranscriptRound(
            agent_uid="causal-analyst",
            user_prompt="Assess whether the treatment effect is identifiable.",
            agent_output="The effect is identifiable if the assumptions hold.",
        )
    )
    state.transcript_history.append(
        TranscriptRound(
            agent_uid="decision-analyst",
            user_prompt="Compare the expected utility of the available actions.",
            agent_output="Action A has the higher expected utility.",
        )
    )

    workspace = _Workspace(
        {
            "memory": [
                "literature/robustness.md",
                "prior_notes.md",
            ],
            "scratchpad": [
                "bayesian-analyst.md",
                "decision-analyst.md",
            ],
            "config": [
                "agent_engine_map.json",
                "agents.json",
                "engines.json",
            ],
        }
    )
    orchestrator = SimpleNamespace(
        ai_team=team,
        llm_pool=_Pool(["gpt-analysis", "gemini-analysis"]),
        interaction_state=state,
    )
    return CLICompleter(workspace, orchestrator)  # type: ignore[arg-type]


def _completions(completer: CLICompleter, text: str) -> list[Completion]:
    document = Document(text=text, cursor_position=len(text))
    event = CompleteEvent(completion_requested=True)
    return list(completer.get_completions(document, event))


def _texts(completer: CLICompleter, text: str) -> list[str]:
    return [completion.text for completion in _completions(completer, text)]


def test_cli_completer_ignores_prompt_text() -> None:
    completer = _completer()

    assert (
        _completions(
            completer,
            "Estimate the posterior probability under the baseline prior.",
        )
        == []
    )


def test_cli_completer_root_commands() -> None:
    completer = _completer()

    completions = _texts(completer, "/")

    assert completions == sorted(completions)
    assert "/agent" in completions
    assert "/engine" in completions
    assert "/history" in completions
    assert "/memory" in completions
    assert "/provider" in completions
    assert "/quit" in completions
    assert "/exit" in completions


def test_cli_completer_fragment() -> None:
    completer = _completer()

    completions = _completions(completer, "/hist")

    assert [completion.text for completion in completions] == [
        "/history",
        "/history-limit",
    ]
    assert all(completion.start_position == -5 for completion in completions)


def test_cli_completer_help_groups() -> None:
    completer = _completer()

    completions = _texts(completer, "/help ")

    assert completions == [
        "agent",
        "engine",
        "files",
        "general",
        "mapping",
        "session",
        "transcript",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "/agent show ",
        "/agent select ",
        "/agent remove ",
        "/mapping show ",
        "/mapping unassign ",
        "/scratchpad show ",
        "/scratchpad append ",
        "/scratchpad clear ",
        "/provider clear ",
    ],
)
def test_cli_completer_agent_uids(text: str) -> None:
    completer = _completer()

    assert _texts(completer, text) == [
        "bayesian-analyst",
        "decision-analyst",
    ]


def test_cli_completer_agent_remove_scratchpad() -> None:
    completer = _completer()

    assert _texts(
        completer,
        "/agent remove bayesian-analyst ",
    ) == ["scratchpad"]
    assert (
        _texts(
            completer,
            "/agent remove causal-analyst ",
        )
        == []
    )


@pytest.mark.parametrize(
    "text",
    [
        "/engine show ",
        "/engine remove ",
    ],
)
def test_cli_completer_engine_uids(text: str) -> None:
    completer = _completer()

    assert _texts(completer, text) == [
        "gemini-analysis",
        "gpt-analysis",
    ]


def test_cli_completer_mapping_assign() -> None:
    completer = _completer()

    assert _texts(completer, "/mapping assign ") == [
        "bayesian-analyst",
        "decision-analyst",
    ]
    assert _texts(
        completer,
        "/mapping assign bayesian-analyst ",
    ) == [
        "gemini-analysis",
        "gpt-analysis",
    ]
    assert (
        _texts(
            completer,
            "/mapping assign causal-analyst ",
        )
        == []
    )


def test_cli_completer_files() -> None:
    completer = _completer()

    assert _texts(completer, "/files ") == [
        "config",
        "memory",
        "scratchpad",
    ]
    assert _texts(completer, "/file ") == [
        "config/agent_engine_map.json",
        "config/agents.json",
        "config/engines.json",
        "memory/literature/robustness.md",
        "memory/prior_notes.md",
        "scratchpad/bayesian-analyst.md",
        "scratchpad/decision-analyst.md",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "/memory show ",
        "/memory append ",
        "/memory clear ",
        "/memory remove ",
    ],
)
def test_cli_completer_memory_files(text: str) -> None:
    completer = _completer()

    assert _texts(completer, text) == [
        "literature/robustness.md",
        "prior_notes.md",
    ]


def test_cli_completer_memory_selection() -> None:
    completer = _completer()

    assert _texts(completer, "/memory select ") == [
        "all",
        "literature/robustness.md",
        "none",
        "prior_notes.md",
    ]
    assert _texts(completer, "/memory select prior_notes.md ") == [
        "literature/robustness.md"
    ]
    assert _texts(completer, "/memory select none ") == []


def test_cli_completer_scratchpad_selection() -> None:
    completer = _completer()

    assert _texts(completer, "/scratchpad select-next ") == [
        "@active",
        "@none",
        "bayesian-analyst",
        "decision-analyst",
    ]
    assert _texts(completer, "/scratchpad select-next @none ") == []
    assert _texts(completer, "/scratchpad select-next decision-analyst ") == [
        "@active",
        "bayesian-analyst",
    ]


def test_cli_completer_context_sections() -> None:
    completer = _completer()

    assert _texts(completer, "/context ") == [
        "identity",
        "memory",
        "scratchpad",
        "system",
        "team",
    ]


def test_cli_completer_history_options() -> None:
    completer = _completer()

    assert _texts(completer, "/history ") == [
        "agent",
        "agents",
        "last",
        "metadata",
        "turns",
    ]


def test_cli_completer_history_turns() -> None:
    completer = _completer()

    assert _texts(completer, "/history turns ") == ["1", "2", "3"]
    assert _texts(completer, "/history turns 1 ") == [
        "2",
        "3",
        "metadata",
    ]
    assert _texts(completer, "/history turns 1 metadata ") == []


def test_cli_completer_history_agent() -> None:
    completer = _completer()

    assert _texts(completer, "/history agent ") == [
        "bayesian-analyst",
        "causal-analyst",
        "decision-analyst",
    ]
    assert _texts(
        completer,
        "/history agent causal-analyst ",
    ) == [
        "last",
        "metadata",
    ]


def test_cli_completer_history_agents() -> None:
    completer = _completer()

    assert _texts(completer, "/history agents ") == [
        "bayesian-analyst",
        "causal-analyst",
        "decision-analyst",
    ]
    assert _texts(
        completer,
        "/history agents bayesian-analyst ",
    ) == [
        "causal-analyst",
        "decision-analyst",
        "last",
        "metadata",
    ]
    assert _texts(
        completer,
        "/history agents bayesian-analyst causal-analyst ",
    ) == [
        "decision-analyst",
        "last",
        "metadata",
    ]
    assert (
        _texts(
            completer,
            "/history agents bayesian-analyst last ",
        )
        == []
    )


def test_cli_completer_turns() -> None:
    completer = _completer()

    assert _texts(completer, "/turn ") == ["1", "2", "3"]
    assert _texts(completer, "/turn 2 ") == ["metadata"]


def test_cli_completer_session_options() -> None:
    completer = _completer()

    assert _texts(completer, "/history-limit ") == ["0", "all"]
    assert _texts(completer, "/state ") == [
        "clear-all",
        "clear-history",
        "clear-provider",
    ]
    assert _texts(completer, "/provider ") == [
        "clear",
        "status",
    ]


def test_cli_completer_history_agent_uids() -> None:
    completer = _completer()

    assert completer._history_agent_uids() == [
        "bayesian-analyst",
        "causal-analyst",
        "decision-analyst",
    ]


def test_cli_completer_turn_range() -> None:
    completer = _completer()

    assert list(completer._turns()) == [1, 2, 3]
