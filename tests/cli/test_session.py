from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from konen import AIAgent, AITeam, EngineRouter, Workspace
from konen.cli.completer import CLICompleter
from konen.cli.session import CLISession


class _ProviderHistory:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def clear_for(self, agent_uid: str, engine_uid: str) -> None:
        self.calls.append(("clear_for", agent_uid, engine_uid))


class _InteractionState:
    def __init__(self) -> None:
        self.provider_history = _ProviderHistory()
        self.calls: list[tuple[object, ...]] = []

    def clear_provider_history(self) -> None:
        self.calls.append(("clear_provider_history",))

    def clear_history(self) -> None:
        self.calls.append(("clear_history",))

    def clear(self) -> None:
        self.calls.append(("clear",))


class _Orchestrator:
    def __init__(self, agent_uid: str | None = "bayesian-analyst") -> None:
        self.ai_team = AITeam(
            [
                AIAgent(
                    name="Bayesian analyst",
                    description="Evaluates uncertainty using Bayesian inference.",
                    uid="bayesian-analyst",
                ),
                AIAgent(
                    name="Decision analyst",
                    description=(
                        "Compares actions using expected utility and "
                        "decision-theoretic trade-offs."
                    ),
                    uid="decision-analyst",
                ),
            ]
        )
        self.engine_router = EngineRouter(
            {
                "bayesian-analyst": "gpt-analysis",
                "decision-analyst": "gemini-analysis",
            }
        )
        self.interaction_state = _InteractionState()
        self.agent_uid = agent_uid
        self.calls: list[tuple[object, ...]] = []
        self.fail_next_turn = False

    def run_turn(
        self,
        user_prompt: str,
        history_limit: int | None = None,
        memory_filenames: tuple[str, ...] | None = None,
        scratchpad_agent_uids: tuple[str, ...] | None = None,
    ) -> str:
        self.calls.append(
            (
                "run_turn",
                user_prompt,
                history_limit,
                memory_filenames,
                scratchpad_agent_uids,
            )
        )
        if self.fail_next_turn:
            self.fail_next_turn = False
            raise RuntimeError("Generation failed.")
        return "The posterior probability remains above the decision threshold."


class _Console:
    def __init__(self) -> None:
        self.status_messages: list[str] = []

    def status(self, message: str) -> nullcontext[None]:
        self.status_messages.append(message)
        return nullcontext()


class _Renderer:
    def __init__(self) -> None:
        self.console = _Console()
        self.calls: list[tuple[object, ...]] = []

    def show_status(self, orchestrator: object) -> None:
        self.calls.append(("status", orchestrator))

    def show_info(self, message: str) -> None:
        self.calls.append(("info", message))

    def show_state(self, state: object) -> None:
        self.calls.append(("state", state))

    def show_provider_status(self, history: object) -> None:
        self.calls.append(("provider_status", history))

    def show_response(self, orchestrator: object, response: str | None) -> None:
        self.calls.append(("response", orchestrator, response))

    def show_agent_selected(self, orchestrator: object, agent_uid: str) -> None:
        self.calls.append(("agent_selected", orchestrator, agent_uid))


class _TestSession(CLISession):
    def __init__(self, agent_uid: str | None = "bayesian-analyst") -> None:
        self.workspace = SimpleNamespace(name="bayesian-decision-study")
        self.orchestrator = _Orchestrator(agent_uid=agent_uid)
        self.renderer = _Renderer()
        self.memory_selections: dict[str, tuple[str, ...] | None] = {}
        self.next_scratchpad_uids: tuple[str, ...] | None = None
        self.history_limit: int | None = None
        self.confirm_answers: list[bool] = []
        self.confirm_messages: list[str] = []

    def _confirm(self, message: str) -> bool:
        self.confirm_messages.append(message)
        if not self.confirm_answers:
            raise AssertionError("No confirmation answer was configured.")
        return self.confirm_answers.pop(0)


def _session_with_workspace(tmp_path: Path) -> _TestSession:
    session = _TestSession()
    workspace = Workspace(tmp_path / "bayesian-decision-study")
    workspace.initialize()
    workspace.add_agent(
        agent_uid="bayesian-analyst",
        name="Bayesian analyst",
        description="Evaluates uncertainty using Bayesian inference.",
    )
    workspace.add_agent(
        agent_uid="decision-analyst",
        name="Decision analyst",
        description="Compares actions using expected utility.",
    )
    workspace.append_to_memory("prior_notes.md", "The baseline prior is 0.40.\n")
    workspace.append_to_memory("posterior_summary.md", "The posterior is 0.82.\n")
    session.workspace = workspace
    return session


def test_cli_session_history_limit_label() -> None:
    session = _TestSession()

    assert session._history_limit_label() == "all"

    session.history_limit = 0
    assert session._history_limit_label() == "0 (no transcript history)"

    session.history_limit = 3
    assert session._history_limit_label() == "3"


def test_cli_session_history_limit_show() -> None:
    session = _TestSession()

    assert session._dispatch_command("/history-limit") is False
    assert session.renderer.calls == [("info", "History limit: all")]


def test_cli_session_history_limit_set() -> None:
    session = _TestSession()

    assert session._dispatch_command("/history-limit 0") is False
    assert session.history_limit == 0
    assert session.renderer.calls[-1] == (
        "info",
        "History limit set to 0 (no transcript history).",
    )

    assert session._dispatch_command("/history-limit 3") is False
    assert session.history_limit == 3
    assert session.renderer.calls[-1] == (
        "info",
        "History limit set to 3.",
    )

    assert session._dispatch_command("/history-limit ALL") is False
    assert session.history_limit is None
    assert session.renderer.calls[-1] == (
        "info",
        "History limit set to all.",
    )


def test_cli_session_status() -> None:
    session = _TestSession()
    session.history_limit = 2

    assert session._dispatch_command("/status") is False
    assert session.renderer.calls == [
        ("status", session.orchestrator),
        ("info", "History limit: 2"),
    ]


def test_cli_session_state_show() -> None:
    session = _TestSession()

    assert session._dispatch_command("/state") is False
    assert session.renderer.calls == [
        ("state", session.orchestrator.interaction_state),
        ("info", "History limit: all"),
    ]


def test_cli_session_state_clear_provider() -> None:
    session = _TestSession()
    session.confirm_answers = [True]

    assert session._dispatch_command("/state clear-provider") is False
    assert session.confirm_messages == ["Clear all provider continuation state?"]
    assert session.orchestrator.interaction_state.calls == [("clear_provider_history",)]
    assert session.renderer.calls == [("info", "Provider continuation state cleared.")]


def test_cli_session_state_clear_provider_cancelled() -> None:
    session = _TestSession()
    session.confirm_answers = [False]

    session._dispatch_command("/state clear-provider")

    assert session.orchestrator.interaction_state.calls == []
    assert session.renderer.calls == [("info", "Provider-state clear cancelled.")]


def test_cli_session_state_clear_history() -> None:
    session = _TestSession()
    session.confirm_answers = [True]

    assert session._dispatch_command("/state clear-history") is False
    assert session.confirm_messages == ["Clear transcript and provider history?"]
    assert session.orchestrator.interaction_state.calls == [("clear_history",)]
    assert session.renderer.calls == [
        ("info", "Transcript and provider history cleared.")
    ]


def test_cli_session_state_clear_all() -> None:
    session = _TestSession()
    session.confirm_answers = [True]

    assert session._dispatch_command("/state clear-all") is False
    assert session.confirm_messages == [
        "Clear the current round, transcript history, and provider state?"
    ]
    assert session.orchestrator.interaction_state.calls == [("clear",)]
    assert session.renderer.calls == [("info", "Interaction state cleared.")]


def test_cli_session_provider_status() -> None:
    session = _TestSession()

    assert session._dispatch_command("/provider status") is False
    assert session.renderer.calls == [
        (
            "provider_status",
            session.orchestrator.interaction_state.provider_history,
        )
    ]


def test_cli_session_provider_clear() -> None:
    session = _TestSession()
    session.confirm_answers = [True]

    assert session._dispatch_command("/provider clear") is False
    assert session.confirm_messages == ["Clear all provider continuation state?"]
    assert session.orchestrator.interaction_state.calls == [("clear_provider_history",)]
    assert session.renderer.calls == [("info", "Provider continuation state cleared.")]


def test_cli_session_provider_clear_agent() -> None:
    session = _TestSession()
    session.confirm_answers = [True]

    assert session._dispatch_command("/provider clear decision-analyst") is False

    assert session.confirm_messages == [
        "Clear provider state for agent 'decision-analyst' with "
        "engine 'gemini-analysis'?"
    ]
    assert session.orchestrator.interaction_state.provider_history.calls == [
        ("clear_for", "decision-analyst", "gemini-analysis")
    ]
    assert session.renderer.calls == [
        (
            "info",
            "Provider state cleared for agent 'decision-analyst' and "
            "engine 'gemini-analysis'.",
        )
    ]


def test_cli_session_provider_clear_agent_cancelled() -> None:
    session = _TestSession()
    session.confirm_answers = [False]

    session._dispatch_command("/provider clear bayesian-analyst")

    assert session.orchestrator.interaction_state.provider_history.calls == []
    assert session.renderer.calls == [("info", "Provider-state clear cancelled.")]


def test_cli_session_agent_select_falls_back() -> None:
    session = _TestSession()

    assert session._dispatch_command("/agent select decision-analyst") is False

    assert session.orchestrator.agent_uid == "decision-analyst"
    assert session.renderer.calls == [
        ("agent_selected", session.orchestrator, "decision-analyst")
    ]


def test_cli_session_memory_selection_per_agent(tmp_path: Path) -> None:
    session = _session_with_workspace(tmp_path)

    session._dispatch_command("/memory select")
    assert session.renderer.calls[-1] == (
        "info",
        "Memory files for agent 'bayesian-analyst': all.",
    )

    session._dispatch_command("/memory select prior_notes.md")
    session._dispatch_command("First question.")
    assert session.orchestrator.calls[-1] == (
        "run_turn",
        "First question.",
        None,
        ("prior_notes.md",),
        None,
    )

    session._dispatch_command("/agent select decision-analyst")
    session._dispatch_command("Second question.")
    assert session.orchestrator.calls[-1] == (
        "run_turn",
        "Second question.",
        None,
        None,
        None,
    )

    session._dispatch_command("/agent select bayesian-analyst")
    session._dispatch_command("Third question.")
    assert session.orchestrator.calls[-1] == (
        "run_turn",
        "Third question.",
        None,
        ("prior_notes.md",),
        None,
    )

    session._dispatch_command(
        "/memory select posterior_summary.md prior_notes.md prior_notes.md"
    )
    session._dispatch_command("Fourth question.")
    assert session.orchestrator.calls[-1] == (
        "run_turn",
        "Fourth question.",
        None,
        ("posterior_summary.md", "prior_notes.md"),
        None,
    )


def test_cli_session_memory_selection_none_and_all(tmp_path: Path) -> None:
    session = _session_with_workspace(tmp_path)

    session._dispatch_command("/memory select none")
    session._dispatch_command("First question.")
    assert session.orchestrator.calls[-1] == (
        "run_turn",
        "First question.",
        None,
        (),
        None,
    )

    session._dispatch_command("/memory select all")
    session._dispatch_command("Second question.")
    assert session.orchestrator.calls[-1] == (
        "run_turn",
        "Second question.",
        None,
        None,
        None,
    )


def test_cli_session_memory_selection_invalid(tmp_path: Path) -> None:
    session = _session_with_workspace(tmp_path)
    session._dispatch_command("/memory select prior_notes.md")

    with pytest.raises(FileNotFoundError, match="Memory file does not exist"):
        session._dispatch_command("/memory select missing.md")
    with pytest.raises(ValueError, match="path escapes memory directory"):
        session._dispatch_command("/memory select ../outside.md")
    with pytest.raises(ValueError, match="Use 'all' or 'none' alone"):
        session._dispatch_command("/memory select all prior_notes.md")

    assert session.memory_selections["bayesian-analyst"] == ("prior_notes.md",)


def test_cli_session_scratchpad_selection(tmp_path: Path) -> None:
    session = _session_with_workspace(tmp_path)

    session._dispatch_command("/scratchpad select-next")
    assert session.renderer.calls[-1] == (
        "info",
        "Scratchpads for next turn: bayesian-analyst (default).",
    )

    session._dispatch_command("/scratchpad select-next @none")
    session._dispatch_command("First question.")
    assert session.orchestrator.calls[-1] == (
        "run_turn",
        "First question.",
        None,
        None,
        (),
    )
    assert session.next_scratchpad_uids is None

    session._dispatch_command(
        "/scratchpad select-next @active decision-analyst @active"
    )
    session._dispatch_command("Second question.")
    assert session.orchestrator.calls[-1] == (
        "run_turn",
        "Second question.",
        None,
        None,
        ("bayesian-analyst", "decision-analyst"),
    )

    session._dispatch_command("/scratchpad select-next decision-analyst")
    session._dispatch_command("Third question.")
    assert session.orchestrator.calls[-1] == (
        "run_turn",
        "Third question.",
        None,
        None,
        ("decision-analyst",),
    )


def test_cli_session_scratchpad_selection_cleared_on_agent_switch(
    tmp_path: Path,
) -> None:
    session = _session_with_workspace(tmp_path)
    session._dispatch_command("/scratchpad select-next @none")

    session._dispatch_command("/agent select decision-analyst")
    session._dispatch_command("First question.")

    assert session.orchestrator.calls[-1] == (
        "run_turn",
        "First question.",
        None,
        None,
        None,
    )


def test_cli_session_scratchpad_selection_retained_after_failure(
    tmp_path: Path,
) -> None:
    session = _session_with_workspace(tmp_path)
    session._dispatch_command("/scratchpad select-next decision-analyst")
    session.orchestrator.fail_next_turn = True

    with pytest.raises(RuntimeError, match="Generation failed"):
        session._dispatch_command("First question.")

    assert session.next_scratchpad_uids == ("decision-analyst",)

    session._dispatch_command("Second question.")
    assert session.orchestrator.calls[-1] == (
        "run_turn",
        "Second question.",
        None,
        None,
        ("decision-analyst",),
    )
    assert session.next_scratchpad_uids is None


def test_cli_session_scratchpad_selection_invalid(tmp_path: Path) -> None:
    session = _session_with_workspace(tmp_path)
    session._dispatch_command("/scratchpad select-next @active")

    with pytest.raises(ValueError, match="Agent 'missing' does not exist"):
        session._dispatch_command("/scratchpad select-next missing")
    with pytest.raises(ValueError, match="Invalid agent UID"):
        session._dispatch_command("/scratchpad select-next @invalid")
    with pytest.raises(ValueError, match="Use '@none' alone"):
        session._dispatch_command("/scratchpad select-next @none @active")

    assert session.next_scratchpad_uids == ("bayesian-analyst",)


@pytest.mark.parametrize("command", ["/memory select", "/scratchpad select-next"])
def test_cli_session_context_selection_no_active_agent(command: str) -> None:
    session = _TestSession(agent_uid=None)

    with pytest.raises(ValueError, match="No active agent"):
        session._dispatch_command(command)


def test_cli_session_run_turn_history_limit() -> None:
    session = _TestSession()
    session.history_limit = 2
    prompt = "Reassess the posterior probability under the wider prior."

    assert session._dispatch_command(prompt) is False

    assert session.orchestrator.calls == [("run_turn", prompt, 2, None, None)]
    assert session.renderer.console.status_messages == ["Generating response..."]
    assert session.renderer.calls == [
        (
            "response",
            session.orchestrator,
            "The posterior probability remains above the decision threshold.",
        )
    ]


def test_cli_session_run_turn_no_agent() -> None:
    session = _TestSession(agent_uid=None)

    with pytest.raises(ValueError, match="No active agent"):
        session._run_turn(
            "Estimate the posterior probability under the baseline prior."
        )

    assert session.orchestrator.calls == []
    assert session.renderer.calls == []


def test_cli_session_completer() -> None:
    session = _TestSession()

    completer = session._completer()

    assert isinstance(completer, CLICompleter)
    assert completer.workspace is session.workspace
    assert completer.orchestrator is session.orchestrator


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("/agents", "Unknown command '/agents'. Use /agent list."),
        (
            "/clear-history",
            "Unknown command '/clear-history'. Use /state clear-history.",
        ),
        (
            "/agent bayesian-analyst",
            "Use /agent select <uid> to select an active agent.",
        ),
    ],
)
def test_cli_session_legacy_command_guard(text: str, message: str) -> None:
    session = _TestSession()

    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        session._dispatch_command(text)
