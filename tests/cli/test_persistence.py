from types import SimpleNamespace

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from konen.cli.app import KonenCLI
from konen.cli.commands import COMMAND_SPECS, static_completions
from konen.cli.completer import CLICompleter


class _Renderer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def show_info(self, message: str) -> None:
        self.calls.append(("info", message))


class _Orchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def append_last_turn_to_memory(self, filename: str) -> None:
        self.calls.append(("memory", filename))

    def append_last_turn_to_scratchpad(self, agent_uid: str) -> None:
        self.calls.append(("scratchpad", agent_uid))


def _cli() -> KonenCLI:
    cli = object.__new__(KonenCLI)
    cli.workspace = SimpleNamespace()
    cli.orchestrator = _Orchestrator()  # type: ignore[assignment]
    cli.renderer = _Renderer()  # type: ignore[assignment]
    return cli


def test_cli_memory_append_turn() -> None:
    cli = _cli()

    cli._handle_memory(["append-turn", "findings.md"])

    assert cli.orchestrator.calls == [("memory", "findings.md")]
    assert cli.renderer.calls == [
        ("info", "Latest transcript turn appended to memory file 'findings.md'.")
    ]


def test_cli_scratchpad_append_turn() -> None:
    cli = _cli()

    cli._handle_scratchpad(["append-turn", "reviewer"])

    assert cli.orchestrator.calls == [("scratchpad", "reviewer")]
    assert cli.renderer.calls == [
        (
            "info",
            "Latest transcript turn appended to scratchpad for agent 'reviewer'.",
        )
    ]


def test_cli_last_turn_command_specs() -> None:
    syntaxes = {spec.syntax for spec in COMMAND_SPECS}
    completions = static_completions()

    assert "/memory append-turn <filename>" in syntaxes
    assert "/scratchpad append-turn <agent_uid>" in syntaxes
    assert "/memory append-turn" in completions
    assert "/scratchpad append-turn" in completions


def test_cli_last_turn_dynamic_completions() -> None:
    workspace = SimpleNamespace(
        list_workspace_files=lambda: {
            "memory": ["findings.md", "notes.md"],
            "scratchpad": ["analyst.md", "reviewer.md"],
            "config": [],
        }
    )
    orchestrator = SimpleNamespace(
        ai_team=SimpleNamespace(list_agents=lambda: ["analyst", "reviewer"]),
        llm_pool=SimpleNamespace(list_engines=lambda: []),
        interaction_state=SimpleNamespace(
            transcript_history=SimpleNamespace(rounds=[])
        ),
    )
    completer = CLICompleter(workspace, orchestrator)  # type: ignore[arg-type]
    event = CompleteEvent(completion_requested=True)

    memory = list(completer.get_completions(Document("/memory append-turn "), event))
    scratchpad = list(
        completer.get_completions(Document("/scratchpad append-turn "), event)
    )

    assert [completion.text for completion in memory] == ["findings.md", "notes.md"]
    assert [completion.text for completion in scratchpad] == ["analyst", "reviewer"]
