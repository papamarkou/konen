from prompt_toolkit.completion import Completer

from .app import KonenCLI
from .completer import CLICompleter


class CLISession(KonenCLI):
    """Konen CLI with transient session controls."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.history_limit: int | None = None

    def _dispatch_command(self, text: str) -> bool:
        """Dispatch session controls before base CLI commands."""
        if not text.startswith("/"):
            self._run_turn(text)
            return False

        parts = text.split()
        command = parts[0]
        arguments = parts[1:]

        if command == "/agents":
            raise ValueError("Unknown command '/agents'. Use /agent list.")

        if command == "/clear-history":
            raise ValueError(
                "Unknown command '/clear-history'. Use /state clear-history."
            )

        if command == "/agent" and len(arguments) == 1:
            if arguments[0] not in {"list", "add"}:
                raise ValueError("Use /agent select <uid> to select an active agent.")

        if command == "/status":
            if arguments:
                raise ValueError("Usage: /status")
            self.renderer.show_status(self.orchestrator)
            self.renderer.show_info(f"History limit: {self._history_limit_label()}")
            return False

        if command == "/history-limit":
            self._handle_history_limit(arguments)
            return False

        if command == "/state":
            self._handle_state(arguments)
            return False

        if command == "/provider":
            self._handle_provider(arguments)
            return False

        return super()._dispatch_command(text)

    def _handle_history_limit(self, arguments: list[str]) -> None:
        """Show or set the history limit for subsequent turns."""
        if not arguments:
            self.renderer.show_info(f"History limit: {self._history_limit_label()}")
            return

        if len(arguments) != 1:
            raise ValueError("Usage: /history-limit [<count>|all]")

        value = arguments[0].lower()
        if value == "all":
            self.history_limit = None
        else:
            self.history_limit = self._parse_non_negative_int(value, "history limit")

        self.renderer.show_info(f"History limit set to {self._history_limit_label()}.")

    def _handle_state(self, arguments: list[str]) -> None:
        """Inspect or clear session interaction state."""
        if not arguments:
            self.renderer.show_state(self.orchestrator.interaction_state)
            self.renderer.show_info(f"History limit: {self._history_limit_label()}")
            return

        if arguments == ["clear-provider"]:
            if not self._confirm("Clear all provider continuation state?"):
                self.renderer.show_info("Provider-state clear cancelled.")
                return
            self.orchestrator.interaction_state.clear_provider_history()
            self.renderer.show_info("Provider continuation state cleared.")
            return

        if arguments == ["clear-history"]:
            self._clear_history()
            return

        if arguments == ["clear-all"]:
            if not self._confirm(
                "Clear the current round, transcript history, and provider state?"
            ):
                self.renderer.show_info("State clear cancelled.")
                return
            self.orchestrator.interaction_state.clear()
            self.renderer.show_info("Interaction state cleared.")
            return

        raise ValueError(
            "Usage: /state | /state clear-provider | "
            "/state clear-history | /state clear-all"
        )

    def _handle_provider(self, arguments: list[str]) -> None:
        """Inspect or clear provider continuation state."""
        if arguments == ["status"]:
            self.renderer.show_provider_status(
                self.orchestrator.interaction_state.provider_history
            )
            return

        if arguments == ["clear"]:
            if not self._confirm("Clear all provider continuation state?"):
                self.renderer.show_info("Provider-state clear cancelled.")
                return
            self.orchestrator.interaction_state.clear_provider_history()
            self.renderer.show_info("Provider continuation state cleared.")
            return

        if len(arguments) == 2 and arguments[0] == "clear":
            agent_uid = arguments[1]
            self.orchestrator.ai_team.get_agent(agent_uid)
            engine_uid = self.orchestrator.engine_router.get_engine_uid(agent_uid)

            if not self._confirm(
                f"Clear provider state for agent '{agent_uid}' with "
                f"engine '{engine_uid}'?"
            ):
                self.renderer.show_info("Provider-state clear cancelled.")
                return

            self.orchestrator.interaction_state.provider_history.clear_for(
                agent_uid=agent_uid, engine_uid=engine_uid
            )
            self.renderer.show_info(
                f"Provider state cleared for agent '{agent_uid}' and "
                f"engine '{engine_uid}'."
            )
            return

        raise ValueError("Usage: /provider status | /provider clear [agent_uid]")

    def _clear_history(self) -> None:
        """Clear transcript history and dependent provider state."""
        if not self._confirm("Clear transcript and provider history?"):
            self.renderer.show_info("History clear cancelled.")
            return

        self.orchestrator.interaction_state.clear_history()
        self.renderer.show_info("Transcript and provider history cleared.")

    def _run_turn(self, user_prompt: str) -> None:
        """Run a turn using the current history limit."""
        if self.orchestrator.agent_uid is None:
            raise ValueError(
                "No active agent. Use /agent select <uid> before sending a prompt."
            )

        with self.renderer.console.status("Generating response..."):
            response = self.orchestrator.run_turn(
                user_prompt, history_limit=self.history_limit
            )

        self.renderer.show_response(orchestrator=self.orchestrator, response=response)

    def _completer(self) -> Completer:
        """Return context-aware completion for the current session."""
        return CLICompleter(workspace=self.workspace, orchestrator=self.orchestrator)

    def _history_limit_label(self) -> str:
        """Return the display label for the current history limit."""
        if self.history_limit is None:
            return "all"
        if self.history_limit == 0:
            return "0 (no transcript history)"
        return str(self.history_limit)
