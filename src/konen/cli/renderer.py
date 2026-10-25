import copy
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from konen import (
    InteractionState,
    LLMPool,
    Orchestrator,
    ProviderHistory,
    Workspace,
)

from .commands import CommandGroup, CommandSpec


class CLIRenderer:
    """Rich-based rendering for the Konen command-line interface."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = Console() if console is None else console

    def show_banner(self) -> None:
        """Render the CLI startup banner."""
        body = Text()
        body.append("Human-centered orchestration of AI agents\n")
        body.append("Use /help for command groups and Tab for completion.")
        self.console.print(Panel(body, title="Konen", expand=False))

    def show_help(
        self, groups: tuple[CommandGroup, ...], commands: tuple[CommandSpec, ...]
    ) -> None:
        """Render a compact index of CLI command groups."""
        table = Table(title="Command groups")
        table.add_column("Group")
        table.add_column("Help")
        table.add_column("Commands", justify="right")

        for group in groups:
            command_count = sum(1 for spec in commands if spec.group == group.key)
            table.add_row(group.title, f"/help {group.key}", str(command_count))

        self.console.print(table)
        self.console.print(
            Text(
                "Use /help <group> for command syntax and descriptions; "
                "press Tab for context-aware completion."
            )
        )

    def show_help_group(
        self, group: CommandGroup, commands: tuple[CommandSpec, ...]
    ) -> None:
        """Render focused help for one command group."""
        table = Table(title=group.title)
        table.add_column("Command")
        table.add_column("Description")

        for spec in commands:
            table.add_row(spec.syntax, spec.description)

        self.console.print(table)

    def show_workspace(self, workspace: Workspace, orchestrator: Orchestrator) -> None:
        """Render workspace paths and configured object counts."""
        files = workspace.list_workspace_files()
        mapping = workspace.load_agent_engine_map_config()["agent_engine_map"]

        table = Table(title="Workspace", show_header=False)
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Root", str(workspace.root_path))
        table.add_row("Memory", str(workspace.memory_path))
        table.add_row("Scratchpad", str(workspace.scratchpad_path))
        table.add_row("Config", str(workspace.config_path))
        table.add_row("Agents", str(len(workspace.list_agents())))
        table.add_row("Engines", str(len(workspace.list_engines())))
        table.add_row("Mappings", str(len(mapping)))
        table.add_row("Memory files", str(len(files["memory"])))
        table.add_row("Scratchpad files", str(len(files["scratchpad"])))
        table.add_row("Config files", str(len(files["config"])))
        table.add_row("Active agent", orchestrator.agent_uid or "—")
        self.console.print(table)

    def show_agents(self, orchestrator: Orchestrator) -> None:
        """Render configured agents."""
        agent_uids = orchestrator.ai_team.list_agents()
        if not agent_uids:
            self.show_info("No agents are configured.")
            return

        table = Table(title="Agents")
        table.add_column("Agent")
        table.add_column("Name")
        table.add_column("Description")

        selected_agent_uid = orchestrator.agent_uid

        for agent_uid in agent_uids:
            agent = orchestrator.ai_team.get_agent(agent_uid)
            display_uid = (
                f"* {agent_uid}" if agent_uid == selected_agent_uid else agent_uid
            )
            table.add_row(display_uid, agent.name, agent.description)

        self.console.print(table)

    def show_agent(self, orchestrator: Orchestrator, agent_uid: str) -> None:
        """Render details for one configured agent."""
        agent = orchestrator.ai_team.get_agent(agent_uid)
        table = Table(title=f"Agent · {agent_uid}", show_header=False)
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Name", agent.name)
        table.add_row("Description", agent.description)
        table.add_row("Active", "yes" if orchestrator.agent_uid == agent_uid else "no")

        try:
            engine_uid = orchestrator.engine_router.get_engine_uid(agent_uid)
            engine = orchestrator.llm_pool.get_engine(engine_uid)
            table.add_row("Engine", engine_uid)
            table.add_row("Model family", engine.model_family)
            table.add_row("Model", engine.model_name)
        except ValueError:
            table.add_row("Engine", "—")

        self.console.print(table)

    def show_agent_selected(self, orchestrator: Orchestrator, agent_uid: str) -> None:
        """Render the newly selected agent."""
        agent = orchestrator.ai_team.get_agent(agent_uid)
        engine_uid = orchestrator.engine_router.get_engine_uid(agent_uid)
        engine = orchestrator.llm_pool.get_engine(engine_uid)

        text = Text()
        text.append(f"{agent.name} ({agent_uid})\n")
        text.append(
            f"Engine: {engine_uid} · {engine.model_family} / {engine.model_name}"
        )
        self.console.print(Panel(text, title="Active agent", expand=False))

    def show_engines(self, llm_pool: LLMPool) -> None:
        """Render configured engines."""
        engine_uids = llm_pool.list_engines()
        if not engine_uids:
            self.show_info("No engines are configured.")
            return

        table = Table(title="Engines")
        table.add_column("Engine")
        table.add_column("Model family")
        table.add_column("Model")
        table.add_column("Config fields")

        for engine_uid in engine_uids:
            engine = llm_pool.get_engine(engine_uid)
            config_fields = ", ".join(sorted(engine.config)) or "—"
            table.add_row(
                engine_uid, engine.model_family, engine.model_name, config_fields
            )

        self.console.print(table)

    def show_engine(self, engine: Any) -> None:
        """Render one engine without exposing secret configuration values."""
        table = Table(title=f"Engine · {engine.uid}", show_header=False)
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Model family", engine.model_family)
        table.add_row("Model", engine.model_name)

        safe_config = self._redact_secrets(engine.config)
        config_text = (
            json.dumps(safe_config, indent=2, ensure_ascii=False)
            if safe_config
            else "{}"
        )
        table.add_row("Config", Text(config_text))
        self.console.print(table)

    def show_mappings(self, orchestrator: Orchestrator) -> None:
        """Render persisted agent-to-engine mappings."""
        agent_uids = orchestrator.ai_team.list_agents()
        if not agent_uids:
            self.show_info("No agents are configured, so there are no mappings.")
            return

        table = Table(title="Agent-engine mappings")
        table.add_column("Agent")
        table.add_column("Agent name")
        table.add_column("Engine")
        table.add_column("Model")

        for agent_uid in agent_uids:
            agent = orchestrator.ai_team.get_agent(agent_uid)
            try:
                engine_uid = orchestrator.engine_router.get_engine_uid(agent_uid)
                engine = orchestrator.llm_pool.get_engine(engine_uid)
                model = engine.model_name
            except ValueError:
                engine_uid = "—"
                model = "—"

            table.add_row(agent_uid, agent.name, engine_uid, model)

        self.console.print(table)

    def show_mapping(
        self, orchestrator: Orchestrator, agent_uid: str, engine_uid: str
    ) -> None:
        """Render one agent-to-engine mapping."""
        agent = orchestrator.ai_team.get_agent(agent_uid)
        engine = orchestrator.llm_pool.get_engine(engine_uid)

        table = Table(title="Agent-engine mapping", show_header=False)
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Agent", agent_uid)
        table.add_row("Agent name", agent.name)
        table.add_row("Engine", engine_uid)
        table.add_row("Model family", engine.model_family)
        table.add_row("Model", engine.model_name)
        self.console.print(table)

    def show_files(
        self, files: dict[str, list[str]], category: str | None = None
    ) -> None:
        """Render managed workspace files."""
        categories = files.keys() if category is None else [category]

        for current_category in categories:
            filenames = files[current_category]
            if not filenames:
                self.show_info(f"No {current_category} files are present.")
                continue

            table = Table(title=f"Files · {current_category}")
            table.add_column("Path")

            for filename in filenames:
                table.add_row(f"{current_category}/{filename}")

            self.console.print(table)

    def show_scratchpads(self, workspace: Workspace) -> None:
        """Render scratchpads for configured agents."""
        filenames = set(workspace.list_workspace_files()["scratchpad"])
        agent_uids = [
            agent_uid
            for agent_uid in workspace.list_agents()
            if f"{agent_uid}.md" in filenames
        ]

        if not agent_uids:
            self.show_info("No agent scratchpads are present.")
            return

        table = Table(title="Files · scratchpad")
        table.add_column("Agent")
        table.add_column("Path")

        for agent_uid in agent_uids:
            table.add_row(agent_uid, f"scratchpad/{agent_uid}.md")

        self.console.print(table)

    def show_file(self, path: str, content: str) -> None:
        """Render one managed workspace text file."""
        suffix = Path(path).suffix.lower()

        if not content:
            renderable: Any = Text("[empty file]")
        elif suffix == ".md":
            renderable = Markdown(content)
        elif suffix == ".json":
            renderable = Syntax(content, "json", word_wrap=True)
        else:
            renderable = Text(content)

        self.console.print(Panel(renderable, title=path))

    def show_response(self, orchestrator: Orchestrator, response: str | None) -> None:
        """Render one agent response."""
        agent_uid = orchestrator.agent_uid
        title = "Agent"

        if agent_uid is not None:
            agent = orchestrator.ai_team.get_agent(agent_uid)
            title = f"{agent.name} · {agent_uid}"

        renderable = (
            Text("[No text output]") if response is None else Markdown(response)
        )
        self.console.print(Panel(renderable, title=title))

    def show_history(self, markdown: str) -> None:
        """Render transcript history markdown."""
        if not markdown:
            self.show_info("Transcript history is empty for this selection.")
            return
        self.console.print(Markdown(markdown))

    def show_context(self, title: str, markdown: str) -> None:
        """Render Markdown interaction context."""
        self.console.print(Panel(Markdown(markdown), title=title))

    def show_text_context(self, title: str, text: str) -> None:
        """Render plain-text interaction context."""
        renderable = Text(text) if text else Text("[empty]")
        self.console.print(Panel(renderable, title=title))

    def show_team_context(self, team_context: dict[str, dict[str, str]]) -> None:
        """Render team context from an interaction round."""
        if not team_context:
            self.show_info("Team context is empty.")
            return

        table = Table(title="Team context")
        table.add_column("Agent")
        table.add_column("Name")
        table.add_column("Description")

        for agent_uid, agent in sorted(team_context.items()):
            table.add_row(
                agent_uid, agent.get("name", ""), agent.get("description", "")
            )

        self.console.print(table)

    def show_identity_context(self, agent_uid: str, identity: dict[str, str]) -> None:
        """Render current-agent identity context."""
        table = Table(title="Identity context", show_header=False)
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Agent", agent_uid)
        table.add_row("Name", identity.get("name", ""))
        table.add_row("Description", identity.get("description", ""))
        self.console.print(table)

    def show_context_files(self, title: str, context: dict[str, str]) -> None:
        """Render file-backed context from an interaction round."""
        if not context:
            self.show_info(f"{title} is empty.")
            return

        self.console.print(Text(title))
        for path, content in sorted(context.items()):
            self.show_file(path, content)

    def show_status(self, orchestrator: Orchestrator) -> None:
        """Render current orchestration status."""
        table = Table(title="Status", show_header=False)
        table.add_column("Field")
        table.add_column("Value")

        agent_uid = orchestrator.agent_uid
        table.add_row("Agent", agent_uid or "—")

        if agent_uid is not None:
            try:
                engine_uid = orchestrator.engine_uid
                engine = orchestrator.llm_pool.get_engine(engine_uid)
                table.add_row("Engine", engine_uid)
                table.add_row("Model family", engine.model_family)
                table.add_row("Model", engine.model_name)
            except ValueError:
                table.add_row("Engine", "—")

        state = orchestrator.interaction_state
        table.add_row("Transcript rounds", str(len(state.transcript_history.rounds)))
        table.add_row("Provider rounds", str(len(state.provider_history.rounds)))
        table.add_row("Current round", self._round_status(state))
        self.console.print(table)

    def show_state(self, state: InteractionState) -> None:
        """Render interaction-state summary."""
        table = Table(title="Interaction state", show_header=False)
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Current round", self._round_status(state))
        table.add_row("Transcript rounds", str(len(state.transcript_history.rounds)))
        table.add_row("Provider rounds", str(len(state.provider_history.rounds)))

        if state.transcript_history.last_round is not None:
            last = state.transcript_history.last_round
            table.add_row("Last transcript UID", last.uid)
            table.add_row("Last agent", last.agent_uid)

        self.console.print(table)

    def show_provider_status(self, history: ProviderHistory) -> None:
        """Render provider-history summary without provider state payloads."""
        if not history.rounds:
            self.show_info("Provider history is empty.")
            return

        table = Table(title="Provider history")
        table.add_column("Transcript UID")
        table.add_column("Agent")
        table.add_column("Engine")
        table.add_column("Model")
        table.add_column("Chain start")

        for round in history.rounds:
            table.add_row(
                round.transcript_uid,
                round.agent_uid,
                round.engine_uid,
                f"{round.model_family} / {round.model_name}",
                round.history_start_uid,
            )

        self.console.print(table)

    def show_info(self, message: str) -> None:
        """Render informational text."""
        self.console.print(Text(message))

    def show_error(self, message: str) -> None:
        """Render an error message."""
        self.console.print(Panel(Text(message), title="Error", expand=False))

    @staticmethod
    def _round_status(state: InteractionState) -> str:
        """Return a concise current-round status."""
        current_round = state.round
        if current_round is None:
            return "none"
        if current_round.transcript.completed_at is not None:
            return "completed"
        return "incomplete"

    @classmethod
    def _redact_secrets(cls, value: Any) -> Any:
        """Return a copy with secret-like dictionary values redacted."""
        if isinstance(value, dict):
            redacted: dict[Any, Any] = {}
            for key, nested_value in value.items():
                key_text = str(key).lower()
                if any(
                    marker in key_text
                    for marker in ("key", "token", "secret", "password")
                ):
                    redacted[key] = "[redacted]"
                else:
                    redacted[key] = cls._redact_secrets(nested_value)
            return redacted

        if isinstance(value, list):
            return [cls._redact_secrets(item) for item in value]

        return copy.deepcopy(value)
