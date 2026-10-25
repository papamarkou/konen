from types import SimpleNamespace

import pytest

from konen.cli.app import KonenCLI


class _Renderer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def show_status(self, orchestrator: object) -> None:
        self.calls.append(("status", orchestrator))

    def show_workspace(self, workspace: object, orchestrator: object) -> None:
        self.calls.append(("workspace", workspace, orchestrator))

    def show_info(self, message: str) -> None:
        self.calls.append(("info", message))

    def show_state(self, state: object) -> None:
        self.calls.append(("state", state))


class _DispatchCLI(KonenCLI):
    def __init__(self) -> None:
        self.workspace = SimpleNamespace(name="bayesian-decision-study")
        self.orchestrator = SimpleNamespace(
            interaction_state=SimpleNamespace(name="interaction-state")
        )
        self.renderer = _Renderer()
        self.calls: list[tuple[object, ...]] = []

    def _run_turn(self, user_prompt: str) -> None:
        self.calls.append(("run_turn", user_prompt))

    def _show_help(self, arguments: list[str]) -> None:
        self.calls.append(("help", list(arguments)))

    def _reload_runtime(self) -> None:
        self.calls.append(("reload",))

    def _handle_agent(self, arguments: list[str]) -> None:
        self.calls.append(("agent", list(arguments)))

    def _handle_engine(self, arguments: list[str]) -> None:
        self.calls.append(("engine", list(arguments)))

    def _handle_mapping(self, arguments: list[str]) -> None:
        self.calls.append(("mapping", list(arguments)))

    def _handle_files(self, arguments: list[str]) -> None:
        self.calls.append(("files", list(arguments)))

    def _handle_file(self, arguments: list[str]) -> None:
        self.calls.append(("file", list(arguments)))

    def _handle_memory(self, arguments: list[str]) -> None:
        self.calls.append(("memory", list(arguments)))

    def _handle_scratchpad(self, arguments: list[str]) -> None:
        self.calls.append(("scratchpad", list(arguments)))

    def _handle_history(self, arguments: list[str]) -> None:
        self.calls.append(("history", list(arguments)))

    def _handle_turn(self, arguments: list[str]) -> None:
        self.calls.append(("turn", list(arguments)))

    def _handle_context(self, arguments: list[str]) -> None:
        self.calls.append(("context", list(arguments)))

    def _handle_provider(self, arguments: list[str]) -> None:
        self.calls.append(("provider", list(arguments)))


def test_cli_dispatch_prompt() -> None:
    cli = _DispatchCLI()
    prompt = "Estimate the posterior probability under the baseline prior."

    should_exit = cli._dispatch_command(prompt)

    assert should_exit is False
    assert cli.calls == [("run_turn", prompt)]


@pytest.mark.parametrize("command", ["/quit", "/exit"])
def test_cli_dispatch_exit(command: str) -> None:
    cli = _DispatchCLI()

    assert cli._dispatch_command(command) is True
    assert cli.calls == []
    assert cli.renderer.calls == []


def test_cli_dispatch_help() -> None:
    cli = _DispatchCLI()

    should_exit = cli._dispatch_command("/help   history")

    assert should_exit is False
    assert cli.calls == [("help", ["history"])]


def test_cli_dispatch_status() -> None:
    cli = _DispatchCLI()

    should_exit = cli._dispatch_command("/status")

    assert should_exit is False
    assert cli.renderer.calls == [("status", cli.orchestrator)]


def test_cli_dispatch_workspace() -> None:
    cli = _DispatchCLI()

    should_exit = cli._dispatch_command("/workspace")

    assert should_exit is False
    assert cli.renderer.calls == [("workspace", cli.workspace, cli.orchestrator)]


def test_cli_dispatch_reload() -> None:
    cli = _DispatchCLI()

    should_exit = cli._dispatch_command("/reload")

    assert should_exit is False
    assert cli.calls == [("reload",)]
    assert cli.renderer.calls == [
        (
            "info",
            "Workspace configuration reloaded; provider state reset.",
        )
    ]


@pytest.mark.parametrize(
    ("text", "handler", "arguments"),
    [
        (
            "/agent show bayesian-analyst",
            "agent",
            ["show", "bayesian-analyst"],
        ),
        (
            "/engine show gpt-analysis",
            "engine",
            ["show", "gpt-analysis"],
        ),
        (
            "/mapping assign bayesian-analyst gpt-analysis",
            "mapping",
            ["assign", "bayesian-analyst", "gpt-analysis"],
        ),
        ("/files memory", "files", ["memory"]),
        (
            "/file memory/prior_notes.md",
            "file",
            ["memory/prior_notes.md"],
        ),
        (
            "/memory show prior_notes.md",
            "memory",
            ["show", "prior_notes.md"],
        ),
        (
            "/scratchpad show bayesian-analyst",
            "scratchpad",
            ["show", "bayesian-analyst"],
        ),
        (
            "/history agent bayesian-analyst last 2 metadata",
            "history",
            ["agent", "bayesian-analyst", "last", "2", "metadata"],
        ),
        ("/turn 2 metadata", "turn", ["2", "metadata"]),
        ("/context memory", "context", ["memory"]),
        ("/provider status", "provider", ["status"]),
    ],
)
def test_cli_dispatch_handler(
    text: str,
    handler: str,
    arguments: list[str],
) -> None:
    cli = _DispatchCLI()

    should_exit = cli._dispatch_command(text)

    assert should_exit is False
    assert cli.calls == [(handler, arguments)]


def test_cli_dispatch_state() -> None:
    cli = _DispatchCLI()

    should_exit = cli._dispatch_command("/state")

    assert should_exit is False
    assert cli.renderer.calls == [("state", cli.orchestrator.interaction_state)]


def test_cli_dispatch_unknown() -> None:
    cli = _DispatchCLI()

    with pytest.raises(ValueError, match="Unknown command '/posterior'"):
        cli._dispatch_command("/posterior")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", {}),
        ('{"reasoning_effort": "high"}', {"reasoning_effort": "high"}),
        (
            '{"temperature": 0.2, "analysis_mode": "bayesian"}',
            {"temperature": 0.2, "analysis_mode": "bayesian"},
        ),
    ],
)
def test_cli_parse_config_json(text: str, expected: dict[str, object]) -> None:
    assert KonenCLI._parse_config_json(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "{invalid-json}",
        '["bayesian", "decision"]',
        '"gpt-5.6-luna"',
    ],
)
def test_cli_parse_config_json_invalid(text: str) -> None:
    with pytest.raises(ValueError, match="Engine config"):
        KonenCLI._parse_config_json(text)


def test_cli_require_argument_count() -> None:
    KonenCLI._require_argument_count(
        ["show", "bayesian-analyst"],
        2,
        "/agent show <uid>",
    )

    with pytest.raises(ValueError, match=r"Usage: /agent show <uid>"):
        KonenCLI._require_argument_count(
            ["show"],
            2,
            "/agent show <uid>",
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", 1), ("3", 3), ("12", 12)],
)
def test_cli_parse_positive_int(value: str, expected: int) -> None:
    assert KonenCLI._parse_positive_int(value, "turn") == expected


@pytest.mark.parametrize("value", ["0", "-1", "second"])
def test_cli_parse_positive_int_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="turn must be a positive integer"):
        KonenCLI._parse_positive_int(value, "turn")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0), ("2", 2), ("10", 10)],
)
def test_cli_parse_non_negative_int(value: str, expected: int) -> None:
    assert KonenCLI._parse_non_negative_int(value, "count") == expected


@pytest.mark.parametrize("value", ["-1", "all"])
def test_cli_parse_non_negative_int_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="count must be a non-negative integer"):
        KonenCLI._parse_non_negative_int(value, "count")


def test_cli_parse_optional_last() -> None:
    assert KonenCLI._parse_optional_last([]) is None
    assert KonenCLI._parse_optional_last(["last", "0"]) == 0
    assert KonenCLI._parse_optional_last(["last", "2"]) == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["last"],
        ["recent", "2"],
        ["last", "2", "metadata"],
        ["last", "-1"],
    ],
)
def test_cli_parse_optional_last_invalid(arguments: list[str]) -> None:
    with pytest.raises(ValueError):
        KonenCLI._parse_optional_last(arguments)


def test_cli_pop_metadata_flag() -> None:
    arguments = ["agent", "bayesian-analyst", "metadata"]

    assert KonenCLI._pop_metadata_flag(arguments) is True
    assert arguments == ["agent", "bayesian-analyst"]


def test_cli_pop_metadata_flag_absent() -> None:
    arguments = ["agent", "bayesian-analyst"]

    assert KonenCLI._pop_metadata_flag(arguments) is False
    assert arguments == ["agent", "bayesian-analyst"]
