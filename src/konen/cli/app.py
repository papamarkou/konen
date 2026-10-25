import json
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, WordCompleter
from prompt_toolkit.history import InMemoryHistory

from konen import Orchestrator, Workspace

from .commands import (
    COMMAND_GROUPS,
    COMMAND_SPECS,
    commands_for_group,
    resolve_group,
)
from .completer import CLICompleter
from .renderer import CLIRenderer


class KonenCLI:
    """Interactive prompt-toolkit command-line interface for Konen."""

    def __init__(
        self,
        workspace: Workspace,
        orchestrator: Orchestrator,
        renderer: CLIRenderer | None = None,
    ) -> None:
        self.workspace = workspace
        self.orchestrator = orchestrator
        self.renderer = CLIRenderer() if renderer is None else renderer
        self.session: PromptSession[str] = PromptSession(history=InMemoryHistory())

    @classmethod
    def from_path(cls, root_path: str | Path = ".") -> "KonenCLI":
        """Build a CLI from a workspace root path."""
        workspace = Workspace(root_path=Path(root_path))
        workspace.initialize()
        orchestrator = Orchestrator.from_workspace(workspace)
        return cls(workspace=workspace, orchestrator=orchestrator)

    def run(self) -> None:
        """Run the interactive CLI until the user exits."""
        self.renderer.show_banner()

        if not self.orchestrator.ai_team.list_agents():
            self.renderer.show_info(
                "No agents are configured. Add agents to the workspace and use /reload."
            )
        elif self.orchestrator.agent_uid is None:
            self.renderer.show_info(
                "No active agent. Use /agent list and /agent select <uid>."
            )

        while True:
            try:
                text = self.session.prompt(
                    self._prompt_text(), completer=self._completer()
                ).strip()
            except (EOFError, KeyboardInterrupt):
                self.renderer.show_info("Exiting Konen.")
                break

            if not text:
                continue

            try:
                if self._dispatch_command(text):
                    break
            except KeyboardInterrupt:
                self._discard_incomplete_round()
                self.renderer.show_info("Operation interrupted.")
            except Exception as exc:
                self._discard_incomplete_round()
                self.renderer.show_error(str(exc))

    def _dispatch_command(self, text: str) -> bool:
        """Dispatch a command or execute one interaction turn."""
        if not text.startswith("/"):
            self._run_turn(text)
            return False

        parts = text.split()
        command = parts[0]
        arguments = parts[1:]

        if command in {"/quit", "/exit"}:
            return True

        if command == "/help":
            self._show_help(arguments)
            return False

        if command == "/status":
            self.renderer.show_status(self.orchestrator)
            return False

        if command == "/workspace":
            self.renderer.show_workspace(
                workspace=self.workspace, orchestrator=self.orchestrator
            )
            return False

        if command == "/reload":
            self._reload_runtime()
            self.renderer.show_info(
                "Workspace configuration reloaded; provider state reset."
            )
            return False

        if command == "/agent":
            self._handle_agent(arguments)
            return False

        if command == "/engine":
            self._handle_engine(arguments)
            return False

        if command == "/mapping":
            self._handle_mapping(arguments)
            return False

        if command == "/files":
            self._handle_files(arguments)
            return False

        if command == "/file":
            self._handle_file(arguments)
            return False

        if command == "/memory":
            self._handle_memory(arguments)
            return False

        if command == "/scratchpad":
            self._handle_scratchpad(arguments)
            return False

        if command == "/history":
            self._handle_history(arguments)
            return False

        if command == "/turn":
            self._handle_turn(arguments)
            return False

        if command == "/context":
            self._handle_context(arguments)
            return False

        if command == "/state":
            self.renderer.show_state(self.orchestrator.interaction_state)
            return False

        if command == "/provider":
            self._handle_provider(arguments)
            return False

        raise ValueError(f"Unknown command '{command}'. Use /help to list commands.")

    def _show_help(self, arguments: list[str]) -> None:
        """Render grouped command help."""
        if len(arguments) > 1:
            raise ValueError("Usage: /help [group]")

        if not arguments:
            self.renderer.show_help(COMMAND_GROUPS, COMMAND_SPECS)
            return

        group = resolve_group(arguments[0])
        if group is None:
            raise ValueError(
                f"Unknown help group '{arguments[0]}'. Use /help to list groups."
            )

        self.renderer.show_help_group(group, commands_for_group(group.key))

    def _handle_agent(self, arguments: list[str]) -> None:
        """Handle agent inspection, selection, and workspace mutations."""
        if not arguments:
            raise ValueError(
                "Usage: /agent list | /agent show <uid> | "
                "/agent select <uid> | /agent add | "
                "/agent remove <uid> [scratchpad]"
            )

        subcommand = arguments[0]

        if subcommand == "list":
            self._require_argument_count(arguments, 1, "/agent list")
            self.renderer.show_agents(self.orchestrator)
            return

        if subcommand == "show":
            self._require_argument_count(arguments, 2, "/agent show <uid>")
            self.renderer.show_agent(
                orchestrator=self.orchestrator, agent_uid=arguments[1]
            )
            return

        if subcommand == "select":
            self._require_argument_count(arguments, 2, "/agent select <uid>")
            self._select_agent(arguments[1])
            return

        if subcommand == "add":
            self._require_argument_count(arguments, 1, "/agent add")
            self._add_agent()
            return

        if subcommand == "remove":
            if len(arguments) not in {2, 3}:
                raise ValueError("Usage: /agent remove <uid> [scratchpad]")
            if len(arguments) == 3 and arguments[2] != "scratchpad":
                raise ValueError("Usage: /agent remove <uid> [scratchpad]")
            self._remove_agent(
                agent_uid=arguments[1], remove_scratchpad=len(arguments) == 3
            )
            return

        raise ValueError(
            "Usage: /agent list | /agent show <uid> | "
            "/agent select <uid> | /agent add | "
            "/agent remove <uid> [scratchpad]"
        )

    def _add_agent(self) -> None:
        """Interactively add an agent to the workspace."""
        agent_uid = self._prompt_required("Agent UID: ")
        name = self._prompt_required("Agent name: ")
        description = self._prompt_required("Agent description: ")

        self.workspace.add_agent(
            agent_uid=agent_uid, name=name, description=description
        )
        self._reload_runtime()
        self.renderer.show_info(f"Agent '{agent_uid}' added.")

    def _remove_agent(self, agent_uid: str, remove_scratchpad: bool) -> None:
        """Remove an agent after explicit confirmation."""
        self.orchestrator.ai_team.get_agent(agent_uid)
        extra = " and delete its scratchpad" if remove_scratchpad else ""
        if not self._confirm(f"Remove agent '{agent_uid}' and its mapping{extra}?"):
            self.renderer.show_info("Agent removal cancelled.")
            return

        self.workspace.remove_agent(
            agent_uid=agent_uid, remove_scratchpad=remove_scratchpad
        )
        self._reload_runtime()
        self.renderer.show_info(f"Agent '{agent_uid}' removed.")

    def _handle_engine(self, arguments: list[str]) -> None:
        """Handle engine inspection and workspace mutations."""
        if not arguments:
            raise ValueError(
                "Usage: /engine list | /engine show <uid> | "
                "/engine add | /engine remove <uid>"
            )

        subcommand = arguments[0]

        if subcommand == "list":
            self._require_argument_count(arguments, 1, "/engine list")
            self.renderer.show_engines(self.orchestrator.llm_pool)
            return

        if subcommand == "show":
            self._require_argument_count(arguments, 2, "/engine show <uid>")
            self.renderer.show_engine(
                self.orchestrator.llm_pool.get_engine(arguments[1])
            )
            return

        if subcommand == "add":
            self._require_argument_count(arguments, 1, "/engine add")
            self._add_engine()
            return

        if subcommand == "remove":
            self._require_argument_count(arguments, 2, "/engine remove <uid>")
            self._remove_engine(arguments[1])
            return

        raise ValueError(
            "Usage: /engine list | /engine show <uid> | "
            "/engine add | /engine remove <uid>"
        )

    def _add_engine(self) -> None:
        """Interactively add an engine to the workspace."""
        engine_uid = self._prompt_required("Engine UID: ")
        model_family = self.session.prompt(
            "Model family: ",
            completer=WordCompleter(
                ["claude", "gemini", "gpt", "kimi"], ignore_case=False
            ),
        ).strip()
        if model_family not in {"claude", "gemini", "gpt", "kimi"}:
            raise ValueError("Model family must be one of: claude, gemini, gpt, kimi.")

        model_name = self._prompt_required("Model name: ")
        config_text = self.session.prompt(
            "Config JSON (optional; do not put API keys here): "
        ).strip()
        config = self._parse_config_json(config_text)

        self.workspace.add_engine(
            engine_uid=engine_uid,
            model_family=model_family,
            model_name=model_name,
            config=config,
        )
        self._reload_runtime()
        self.renderer.show_info(f"Engine '{engine_uid}' added.")

    def _remove_engine(self, engine_uid: str) -> None:
        """Remove an unassigned engine after explicit confirmation."""
        self.orchestrator.llm_pool.get_engine(engine_uid)
        if not self._confirm(f"Remove engine '{engine_uid}'?"):
            self.renderer.show_info("Engine removal cancelled.")
            return

        self.workspace.remove_engine(engine_uid)
        self._reload_runtime()
        self.renderer.show_info(f"Engine '{engine_uid}' removed.")

    def _handle_mapping(self, arguments: list[str]) -> None:
        """Handle agent-engine mapping inspection and mutations."""
        if not arguments:
            raise ValueError(
                "Usage: /mapping list | /mapping show <agent_uid> | "
                "/mapping assign <agent_uid> <engine_uid> | "
                "/mapping unassign <agent_uid>"
            )

        subcommand = arguments[0]

        if subcommand == "list":
            self._require_argument_count(arguments, 1, "/mapping list")
            self.renderer.show_mappings(self.orchestrator)
            return

        if subcommand == "show":
            self._require_argument_count(arguments, 2, "/mapping show <agent_uid>")
            agent_uid = arguments[1]
            engine_uid = self.orchestrator.engine_router.get_engine_uid(agent_uid)
            self.renderer.show_mapping(
                orchestrator=self.orchestrator,
                agent_uid=agent_uid,
                engine_uid=engine_uid,
            )
            return

        if subcommand == "assign":
            self._require_argument_count(
                arguments, 3, "/mapping assign <agent_uid> <engine_uid>"
            )
            self._assign_engine(agent_uid=arguments[1], engine_uid=arguments[2])
            return

        if subcommand == "unassign":
            self._require_argument_count(arguments, 2, "/mapping unassign <agent_uid>")
            self._unassign_engine(arguments[1])
            return

        raise ValueError(
            "Usage: /mapping list | /mapping show <agent_uid> | "
            "/mapping assign <agent_uid> <engine_uid> | "
            "/mapping unassign <agent_uid>"
        )

    def _assign_engine(self, agent_uid: str, engine_uid: str) -> None:
        """Persist an agent-engine assignment and refresh runtime state."""
        self.orchestrator.ai_team.get_agent(agent_uid)
        self.orchestrator.llm_pool.get_engine(engine_uid)

        current_engine_uid = self.orchestrator.engine_router.table.get(agent_uid)
        if (
            current_engine_uid is not None
            and current_engine_uid != engine_uid
            and not self._confirm(
                f"Reassign agent '{agent_uid}' from '{current_engine_uid}' "
                f"to '{engine_uid}'?"
            )
        ):
            self.renderer.show_info("Mapping change cancelled.")
            return

        self.workspace.assign_engine(agent_uid, engine_uid)
        self._reload_runtime()
        self.renderer.show_info(
            f"Agent '{agent_uid}' assigned to engine '{engine_uid}'."
        )

    def _unassign_engine(self, agent_uid: str) -> None:
        """Remove an agent-engine assignment after confirmation."""
        engine_uid = self.orchestrator.engine_router.get_engine_uid(agent_uid)
        if not self._confirm(
            f"Unassign engine '{engine_uid}' from agent '{agent_uid}'?"
        ):
            self.renderer.show_info("Mapping removal cancelled.")
            return

        self.workspace.unassign_engine(agent_uid)
        self._reload_runtime()
        self.renderer.show_info(f"Engine assignment removed from agent '{agent_uid}'.")

    def _handle_files(self, arguments: list[str]) -> None:
        """Handle workspace file listing."""
        if len(arguments) > 1:
            raise ValueError("Usage: /files [memory|scratchpad|config]")

        category = None if not arguments else arguments[0]
        if category not in {None, "memory", "scratchpad", "config"}:
            raise ValueError("Usage: /files [memory|scratchpad|config]")

        files = self.workspace.list_workspace_files()
        self.renderer.show_files(files, category=category)

    def _handle_file(self, arguments: list[str]) -> None:
        """Handle generic managed workspace file viewing."""
        self._require_argument_count(arguments, 1, "/file <workspace-relative-path>")
        path = arguments[0]
        content = self.workspace.read_workspace_file(path)
        self.renderer.show_file(path, content)

    def _handle_memory(self, arguments: list[str]) -> None:
        """Handle memory-file inspection and mutations."""
        if not arguments:
            raise ValueError(
                "Usage: /memory list | /memory show <filename> | "
                "/memory append <filename> | /memory append-turn <filename> | "
                "/memory clear <filename> | /memory remove <filename>"
            )

        subcommand = arguments[0]

        if subcommand == "list":
            self._require_argument_count(arguments, 1, "/memory list")
            files = self.workspace.list_workspace_files()
            self.renderer.show_files(files, category="memory")
            return

        if subcommand == "show":
            self._require_argument_count(arguments, 2, "/memory show <filename>")
            filename = arguments[1]
            path = f"{self.workspace.memory_dir_name}/{filename}"
            content = self.workspace.read_workspace_file(path)
            self.renderer.show_file(path, content)
            return

        if subcommand == "append":
            self._require_argument_count(arguments, 2, "/memory append <filename>")
            filename = arguments[1]
            text = self._prompt_append_text("Memory text: ")
            self.workspace.append_to_memory(filename, text)
            self.renderer.show_info(f"Text appended to memory file '{filename}'.")
            return

        if subcommand == "append-turn":
            self._require_argument_count(arguments, 2, "/memory append-turn <filename>")
            filename = arguments[1]
            self.orchestrator.append_last_turn_to_memory(filename)
            self.renderer.show_info(
                f"Latest transcript turn appended to memory file '{filename}'."
            )
            return

        if subcommand == "clear":
            self._require_argument_count(arguments, 2, "/memory clear <filename>")
            filename = arguments[1]
            if not self._confirm(f"Clear memory file '{filename}'?"):
                self.renderer.show_info("Memory clear cancelled.")
                return
            self.workspace.clear_memory_file(filename)
            self.renderer.show_info(f"Memory file '{filename}' cleared.")
            return

        if subcommand == "remove":
            self._require_argument_count(arguments, 2, "/memory remove <filename>")
            filename = arguments[1]
            if not self._confirm(f"Remove memory file '{filename}'?"):
                self.renderer.show_info("Memory removal cancelled.")
                return
            self.workspace.remove_memory_file(filename)
            self.renderer.show_info(f"Memory file '{filename}' removed.")
            return

        raise ValueError(
            "Usage: /memory list | /memory show <filename> | "
            "/memory append <filename> | /memory append-turn <filename> | "
            "/memory clear <filename> | /memory remove <filename>"
        )

    def _handle_scratchpad(self, arguments: list[str]) -> None:
        """Handle scratchpad inspection and mutations."""
        if not arguments:
            raise ValueError(
                "Usage: /scratchpad list | /scratchpad show <agent_uid> | "
                "/scratchpad append <agent_uid> | "
                "/scratchpad append-turn <agent_uid> | "
                "/scratchpad clear <agent_uid> | /scratchpad clear-all"
            )

        subcommand = arguments[0]

        if subcommand == "list":
            self._require_argument_count(arguments, 1, "/scratchpad list")
            self.renderer.show_scratchpads(self.workspace)
            return

        if subcommand == "show":
            self._require_argument_count(arguments, 2, "/scratchpad show <agent_uid>")
            agent_uid = arguments[1]
            context = self.workspace.load_scratchpad_context(agent_uid)
            path, content = next(iter(context.items()))
            self.renderer.show_file(path, content)
            return

        if subcommand == "append":
            self._require_argument_count(arguments, 2, "/scratchpad append <agent_uid>")
            agent_uid = arguments[1]
            self.orchestrator.ai_team.get_agent(agent_uid)
            text = self._prompt_append_text("Scratchpad text: ")
            self.workspace.append_to_scratchpad(agent_uid, text)
            self.renderer.show_info(
                f"Text appended to scratchpad for agent '{agent_uid}'."
            )
            return

        if subcommand == "append-turn":
            self._require_argument_count(
                arguments, 2, "/scratchpad append-turn <agent_uid>"
            )
            agent_uid = arguments[1]
            self.orchestrator.append_last_turn_to_scratchpad(agent_uid)
            self.renderer.show_info(
                "Latest transcript turn appended to scratchpad for "
                f"agent '{agent_uid}'."
            )
            return

        if subcommand == "clear":
            self._require_argument_count(arguments, 2, "/scratchpad clear <agent_uid>")
            agent_uid = arguments[1]
            self.orchestrator.ai_team.get_agent(agent_uid)
            if not self._confirm(f"Clear scratchpad for agent '{agent_uid}'?"):
                self.renderer.show_info("Scratchpad clear cancelled.")
                return
            self.workspace.clear_scratchpad(agent_uid)
            self.renderer.show_info(f"Scratchpad for agent '{agent_uid}' cleared.")
            return

        if subcommand == "clear-all":
            self._require_argument_count(arguments, 1, "/scratchpad clear-all")
            if not self._confirm("Clear all configured agent scratchpads?"):
                self.renderer.show_info("Scratchpad clear-all cancelled.")
                return
            self.workspace.clear_all_scratchpads()
            self.renderer.show_info("All configured agent scratchpads cleared.")
            return

        raise ValueError(
            "Usage: /scratchpad list | /scratchpad show <agent_uid> | "
            "/scratchpad append <agent_uid> | "
            "/scratchpad append-turn <agent_uid> | "
            "/scratchpad clear <agent_uid> | /scratchpad clear-all"
        )

    def _handle_history(self, arguments: list[str]) -> None:
        """Handle transcript history inspection commands."""
        history = self.orchestrator.interaction_state.transcript_history

        if not arguments:
            self.renderer.show_history(history.to_markdown())
            return

        include_metadata = self._pop_metadata_flag(arguments)
        if not arguments:
            self.renderer.show_history(
                history.to_markdown(include_metadata=include_metadata)
            )
            return

        subcommand = arguments[0]

        if subcommand == "last":
            self._require_argument_count(
                arguments, 2, "/history last <count> [metadata]"
            )
            count = self._parse_non_negative_int(arguments[1], "count")
            markdown = history.to_markdown(
                history_limit=count, include_metadata=include_metadata
            )
            self.renderer.show_history(markdown)
            return

        if subcommand == "turns":
            if len(arguments) < 2:
                raise ValueError("Usage: /history turns <turn> [<turn> ...] [metadata]")
            turns = [self._parse_positive_int(value, "turn") for value in arguments[1:]]
            markdown = history.to_markdown(
                turns=turns, include_metadata=include_metadata
            )
            self.renderer.show_history(markdown)
            return

        if subcommand == "agent":
            self._show_agent_history(arguments[1:], include_metadata=include_metadata)
            return

        if subcommand == "agents":
            self._show_agents_history(arguments[1:], include_metadata=include_metadata)
            return

        raise ValueError(
            "Usage: /history | /history last <count> | "
            "/history turns <turn> [...] | /history agent <uid> [...] | "
            "/history agents <uid> [...]"
        )

    def _show_agent_history(self, arguments: list[str], include_metadata: bool) -> None:
        """Render global transcript turns belonging to one agent."""
        if not arguments:
            raise ValueError("Usage: /history agent <uid> [last <count>] [metadata]")

        agent_uid = arguments[0]
        self._require_history_agent_present(agent_uid)
        remaining = arguments[1:]
        limit = self._parse_optional_last(remaining)
        turns = self._global_turns_for_agents({agent_uid})

        if limit is not None:
            turns = turns[-limit:] if limit else []

        markdown = self.orchestrator.interaction_state.transcript_history.to_markdown(
            turns=turns, include_metadata=include_metadata
        )
        self.renderer.show_history(markdown)

    def _show_agents_history(
        self, arguments: list[str], include_metadata: bool
    ) -> None:
        """Render global transcript turns belonging to selected agents."""
        if not arguments:
            raise ValueError(
                "Usage: /history agents <uid> [<uid> ...] [last <count>] [metadata]"
            )

        if "last" in arguments:
            last_index = arguments.index("last")
            agent_uids = arguments[:last_index]
            remaining = arguments[last_index:]
        else:
            agent_uids = arguments
            remaining = []

        if not agent_uids:
            raise ValueError("At least one agent UID is required before 'last'.")

        for agent_uid in agent_uids:
            self._require_history_agent_present(agent_uid)

        limit = self._parse_optional_last(remaining)
        turns = self._global_turns_for_agents(set(agent_uids))

        if limit is not None:
            turns = turns[-limit:] if limit else []

        markdown = self.orchestrator.interaction_state.transcript_history.to_markdown(
            turns=turns, include_metadata=include_metadata
        )
        self.renderer.show_history(markdown)

    def _handle_turn(self, arguments: list[str]) -> None:
        """Handle one-turn transcript inspection."""
        include_metadata = self._pop_metadata_flag(arguments)
        self._require_argument_count(arguments, 1, "/turn <turn> [metadata]")
        turn = self._parse_positive_int(arguments[0], "turn")
        history = self.orchestrator.interaction_state.transcript_history
        markdown = history.to_markdown(turns=[turn], include_metadata=include_metadata)
        self.renderer.show_history(markdown)

    def _handle_context(self, arguments: list[str]) -> None:
        """Handle latest interaction-context inspection."""
        if len(arguments) > 1:
            raise ValueError("Usage: /context [system|team|identity|memory|scratchpad]")

        interaction_round = self.orchestrator.interaction_state.round
        if interaction_round is None:
            raise ValueError("No interaction round is available.")

        section = None if not arguments else arguments[0]
        if section is None:
            self.renderer.show_context(
                "Prompt context", interaction_round.build_prompt_context()
            )
            return

        if section == "system":
            self.renderer.show_text_context(
                "System context", interaction_round.system_context
            )
            return

        if section == "team":
            self.renderer.show_team_context(interaction_round.team_context)
            return

        if section == "identity":
            self.renderer.show_identity_context(
                agent_uid=interaction_round.agent_uid,
                identity=interaction_round.identity_context,
            )
            return

        if section == "memory":
            self.renderer.show_context_files(
                "Memory context", interaction_round.memory_context
            )
            return

        if section == "scratchpad":
            self.renderer.show_context_files(
                "Scratchpad context", interaction_round.scratchpad_context
            )
            return

        raise ValueError("Usage: /context [system|team|identity|memory|scratchpad]")

    def _handle_provider(self, arguments: list[str]) -> None:
        """Handle read-only provider-history commands."""
        if arguments != ["status"]:
            raise ValueError("Usage: /provider status")

        self.renderer.show_provider_status(
            self.orchestrator.interaction_state.provider_history
        )

    def _run_turn(self, user_prompt: str) -> None:
        """Execute and render one orchestrator turn."""
        if self.orchestrator.agent_uid is None:
            raise ValueError(
                "No active agent. Use /agent select <uid> before sending a prompt."
            )

        with self.renderer.console.status("Generating response..."):
            response = self.orchestrator.run_turn(user_prompt)

        self.renderer.show_response(orchestrator=self.orchestrator, response=response)

    def _select_agent(self, agent_uid: str) -> None:
        """Select one configured agent for subsequent turns."""
        self.orchestrator.ai_team.get_agent(agent_uid)
        self.orchestrator.engine_router.get_engine_uid(agent_uid)
        self.orchestrator.agent_uid = agent_uid
        self.renderer.show_agent_selected(
            orchestrator=self.orchestrator, agent_uid=agent_uid
        )

    def _reload_runtime(self) -> None:
        """Reload runtime configuration and reset provider continuation state."""
        self.orchestrator.reload_from_workspace()
        self.orchestrator.interaction_state.clear_provider_history()

    def _history_agent_uids(self) -> set[str]:
        """Return agent UIDs known from current configuration or transcript."""
        agent_uids = set(self.orchestrator.ai_team.list_agents())
        agent_uids.update(
            round.agent_uid
            for round in self.orchestrator.interaction_state.transcript_history.rounds
        )
        return agent_uids

    def _require_history_agent_present(self, agent_uid: str) -> None:
        """Require an agent UID known from configuration or transcript."""
        if agent_uid not in self._history_agent_uids():
            raise ValueError(
                f"Agent '{agent_uid}' is not configured and does not appear "
                "in transcript history."
            )

    def _global_turns_for_agents(self, agent_uids: set[str]) -> list[int]:
        """Return global 1-based transcript turns for selected agents."""
        history = self.orchestrator.interaction_state.transcript_history
        return [
            turn
            for turn, round in enumerate(history.rounds, start=1)
            if round.agent_uid in agent_uids
        ]

    def _prompt_text(self) -> str:
        """Return the current prompt label."""
        agent_uid = self.orchestrator.agent_uid
        if agent_uid is None:
            return "konen> "
        return f"konen[{agent_uid}]> "

    def _completer(self) -> Completer:
        """Return live context-aware completion for the current CLI state."""
        return CLICompleter(workspace=self.workspace, orchestrator=self.orchestrator)

    def _discard_incomplete_round(self) -> None:
        """Discard a failed current round so the REPL can continue."""
        current_round = self.orchestrator.interaction_state.round
        if current_round is not None and current_round.transcript.completed_at is None:
            self.orchestrator.interaction_state.clear_round()

    def _prompt_required(self, label: str) -> str:
        """Prompt for a required non-empty value."""
        value = self.session.prompt(label).strip()
        if not value:
            raise ValueError(f"{label.rstrip(': ')} must not be empty.")
        return value

    def _prompt_append_text(self, label: str) -> str:
        """Prompt for one line of text and append a trailing newline."""
        text = self.session.prompt(label)
        if not text:
            raise ValueError("Text must not be empty.")
        return text + "\n"

    def _confirm(self, message: str) -> bool:
        """Request explicit yes/no confirmation for a destructive operation."""
        answer = self.session.prompt(f"{message} [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        raise ValueError("Confirmation must be 'y', 'yes', 'n', or 'no'.")

    @staticmethod
    def _parse_config_json(text: str) -> dict[str, object]:
        """Parse optional engine configuration JSON as an object."""
        if not text:
            return {}

        try:
            config = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Engine config must be valid JSON.") from exc

        if not isinstance(config, dict):
            raise ValueError("Engine config JSON must be an object.")

        return config

    @staticmethod
    def _require_argument_count(
        arguments: list[str], expected: int, usage: str
    ) -> None:
        """Require an exact argument count for one command form."""
        if len(arguments) != expected:
            raise ValueError(f"Usage: {usage}")

    @staticmethod
    def _parse_positive_int(value: str, label: str) -> int:
        """Parse a positive integer CLI argument."""
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a positive integer.") from exc

        if parsed <= 0:
            raise ValueError(f"{label} must be a positive integer.")

        return parsed

    @staticmethod
    def _parse_non_negative_int(value: str, label: str) -> int:
        """Parse a non-negative integer CLI argument."""
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a non-negative integer.") from exc

        if parsed < 0:
            raise ValueError(f"{label} must be a non-negative integer.")

        return parsed

    @staticmethod
    def _parse_optional_last(arguments: list[str]) -> int | None:
        """Parse an optional 'last <count>' suffix."""
        if not arguments:
            return None

        if len(arguments) != 2 or arguments[0] != "last":
            raise ValueError("Expected optional suffix: last <count>")

        return KonenCLI._parse_non_negative_int(arguments[1], "count")

    @staticmethod
    def _pop_metadata_flag(arguments: list[str]) -> bool:
        """Remove and return a trailing metadata display flag."""
        if arguments and arguments[-1] == "metadata":
            arguments.pop()
            return True
        return False
