from types import SimpleNamespace

import pytest

from konen.cli.app import KonenCLI


class _ScriptedSession:
    def __init__(self, actions: list[object]) -> None:
        self.actions = list(actions)
        self.calls: list[tuple[str, object | None]] = []

    def prompt(self, label: str, completer: object | None = None) -> str:
        self.calls.append((label, completer))
        if not self.actions:
            raise AssertionError("No scripted prompt action remains.")

        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        if not isinstance(action, str):
            raise AssertionError(
                "Scripted prompt actions must be strings or exceptions."
            )
        return action


class _Team:
    def __init__(self, agent_uids: list[str]) -> None:
        self.agent_uids = list(agent_uids)

    def list_agents(self) -> list[str]:
        return list(self.agent_uids)


class _InteractionState:
    def __init__(self) -> None:
        self.round: object | None = None
        self.clear_round_calls = 0

    def clear_round(self) -> None:
        self.clear_round_calls += 1
        self.round = None


class _Renderer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def show_banner(self) -> None:
        self.calls.append(("banner",))

    def show_info(self, message: str) -> None:
        self.calls.append(("info", message))

    def show_error(self, message: str) -> None:
        self.calls.append(("error", message))


class _LifecycleCLI(KonenCLI):
    def __init__(
        self,
        actions: list[object],
        *,
        agent_uids: list[str] | None = None,
        agent_uid: str | None = "bayesian-analyst",
    ) -> None:
        configured_agents = (
            ["bayesian-analyst", "decision-analyst"]
            if agent_uids is None
            else agent_uids
        )
        self.workspace = SimpleNamespace()
        self.orchestrator = SimpleNamespace(
            ai_team=_Team(configured_agents),
            agent_uid=agent_uid,
            interaction_state=_InteractionState(),
        )
        self.renderer = _Renderer()
        self.session = _ScriptedSession(actions)
        self.dispatch_calls: list[str] = []
        self.completer = object()

    def _dispatch_command(self, text: str) -> bool:
        self.dispatch_calls.append(text)

        if text == "/quit":
            return True

        if text == "/fail":
            self.orchestrator.interaction_state.round = SimpleNamespace(
                transcript=SimpleNamespace(completed_at=None)
            )
            raise ValueError("The selected agent has no engine mapping.")

        if text == "/interrupt":
            self.orchestrator.interaction_state.round = SimpleNamespace(
                transcript=SimpleNamespace(completed_at=None)
            )
            raise KeyboardInterrupt

        return False

    def _completer(self) -> object:
        return self.completer


def test_cli_run_banner_and_exit() -> None:
    cli = _LifecycleCLI(["/quit"])

    cli.run()

    assert cli.renderer.calls == [("banner",)]
    assert cli.dispatch_calls == ["/quit"]
    assert cli.session.calls == [
        ("konen[bayesian-analyst]> ", cli.completer),
    ]


def test_cli_run_no_agents_notice() -> None:
    cli = _LifecycleCLI(
        ["/quit"],
        agent_uids=[],
        agent_uid=None,
    )

    cli.run()

    assert cli.renderer.calls == [
        ("banner",),
        (
            "info",
            "No agents are configured. Add agents to the workspace and use /reload.",
        ),
    ]


def test_cli_run_no_active_agent_notice() -> None:
    cli = _LifecycleCLI(["/quit"], agent_uid=None)

    cli.run()

    assert cli.renderer.calls == [
        ("banner",),
        (
            "info",
            "No active agent. Use /agent list and /agent select <uid>.",
        ),
    ]
    assert cli.session.calls == [("konen> ", cli.completer)]


def test_cli_run_skips_blank_input() -> None:
    cli = _LifecycleCLI(
        [
            "   ",
            "Estimate the posterior probability under the baseline prior.  ",
            "/quit",
        ]
    )

    cli.run()

    assert cli.dispatch_calls == [
        "Estimate the posterior probability under the baseline prior.",
        "/quit",
    ]
    assert len(cli.session.calls) == 3


@pytest.mark.parametrize("exc", [EOFError(), KeyboardInterrupt()])
def test_cli_run_prompt_exit(exc: BaseException) -> None:
    cli = _LifecycleCLI([exc])

    cli.run()

    assert cli.dispatch_calls == []
    assert cli.renderer.calls == [
        ("banner",),
        ("info", "Exiting Konen."),
    ]


def test_cli_run_dispatch_error_recovers() -> None:
    cli = _LifecycleCLI(["/fail", "/quit"])

    cli.run()

    state = cli.orchestrator.interaction_state
    assert cli.dispatch_calls == ["/fail", "/quit"]
    assert state.clear_round_calls == 1
    assert state.round is None
    assert cli.renderer.calls == [
        ("banner",),
        ("error", "The selected agent has no engine mapping."),
    ]


def test_cli_run_dispatch_interrupt_recovers() -> None:
    cli = _LifecycleCLI(["/interrupt", "/quit"])

    cli.run()

    state = cli.orchestrator.interaction_state
    assert cli.dispatch_calls == ["/interrupt", "/quit"]
    assert state.clear_round_calls == 1
    assert state.round is None
    assert cli.renderer.calls == [
        ("banner",),
        ("info", "Operation interrupted."),
    ]


def test_cli_prompt_text() -> None:
    cli = _LifecycleCLI(["/quit"])

    assert cli._prompt_text() == "konen[bayesian-analyst]> "

    cli.orchestrator.agent_uid = "decision-analyst"
    assert cli._prompt_text() == "konen[decision-analyst]> "

    cli.orchestrator.agent_uid = None
    assert cli._prompt_text() == "konen> "


def test_cli_discard_incomplete_round() -> None:
    cli = _LifecycleCLI(["/quit"])
    state = cli.orchestrator.interaction_state
    state.round = SimpleNamespace(transcript=SimpleNamespace(completed_at=None))

    cli._discard_incomplete_round()

    assert state.clear_round_calls == 1
    assert state.round is None


def test_cli_discard_completed_round() -> None:
    cli = _LifecycleCLI(["/quit"])
    state = cli.orchestrator.interaction_state
    completed_round = SimpleNamespace(
        transcript=SimpleNamespace(completed_at="2026-08-20T08:00:00+03:00")
    )
    state.round = completed_round

    cli._discard_incomplete_round()

    assert state.clear_round_calls == 0
    assert state.round is completed_round


def test_cli_discard_no_round() -> None:
    cli = _LifecycleCLI(["/quit"])
    state = cli.orchestrator.interaction_state

    cli._discard_incomplete_round()

    assert state.clear_round_calls == 0
    assert state.round is None


def test_cli_prompt_required() -> None:
    cli = _LifecycleCLI(["  Bayesian analyst  "])

    value = cli._prompt_required("Agent name: ")

    assert value == "Bayesian analyst"
    assert cli.session.calls == [("Agent name: ", None)]


def test_cli_prompt_required_empty() -> None:
    cli = _LifecycleCLI(["   "])

    with pytest.raises(ValueError, match="Agent name must not be empty"):
        cli._prompt_required("Agent name: ")


def test_cli_prompt_append_text() -> None:
    cli = _LifecycleCLI(["Compare posterior sensitivity under prior scales 1 and 2."])

    text = cli._prompt_append_text("Memory text: ")

    assert text == ("Compare posterior sensitivity under prior scales 1 and 2.\n")


def test_cli_prompt_append_text_empty() -> None:
    cli = _LifecycleCLI([""])

    with pytest.raises(ValueError, match="Text must not be empty"):
        cli._prompt_append_text("Scratchpad text: ")


@pytest.mark.parametrize("answer", ["y", "yes", "Y", " YES "])
def test_cli_confirm_yes(answer: str) -> None:
    cli = _LifecycleCLI([answer])

    assert cli._confirm("Clear transcript and provider history?") is True


@pytest.mark.parametrize("answer", ["", "n", "no", "N", " NO "])
def test_cli_confirm_no(answer: str) -> None:
    cli = _LifecycleCLI([answer])

    assert cli._confirm("Clear transcript and provider history?") is False


def test_cli_confirm_invalid() -> None:
    cli = _LifecycleCLI(["maybe"])

    with pytest.raises(
        ValueError,
        match="Confirmation must be 'y', 'yes', 'n', or 'no'",
    ):
        cli._confirm("Clear transcript and provider history?")
