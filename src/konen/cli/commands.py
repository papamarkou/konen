from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandGroup:
    """Group of related CLI commands."""

    key: str
    title: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Specification for a CLI command."""

    group: str
    syntax: str
    description: str
    completion: str


COMMAND_GROUPS: tuple[CommandGroup, ...] = (
    CommandGroup("general", "General"),
    CommandGroup("agent", "Agents"),
    CommandGroup("engine", "Engines"),
    CommandGroup("mapping", "Mappings"),
    CommandGroup("files", "Files", aliases=("file", "memory", "scratchpad")),
    CommandGroup("transcript", "Transcript", aliases=("history", "turn", "context")),
    CommandGroup(
        "session", "Session state", aliases=("state", "provider", "history-limit")
    ),
)


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        "general",
        "/help [group]",
        "Show grouped help or detailed help for one group.",
        "/help",
    ),
    CommandSpec(
        "general", "/status", "Show the current interaction status.", "/status"
    ),
    CommandSpec(
        "general",
        "/workspace",
        "Show workspace paths and configured object counts.",
        "/workspace",
    ),
    CommandSpec(
        "general",
        "/reload",
        "Reload workspace configuration and reset provider state.",
        "/reload",
    ),
    CommandSpec("general", "/quit", "Exit Konen.", "/quit"),
    CommandSpec(
        "agent",
        "/agent list",
        "List configured agents and their engine assignments.",
        "/agent list",
    ),
    CommandSpec(
        "agent", "/agent show <uid>", "Show one configured agent.", "/agent show"
    ),
    CommandSpec(
        "agent",
        "/agent select <uid>",
        "Select the active agent for subsequent prompts.",
        "/agent select",
    ),
    CommandSpec(
        "agent", "/agent add", "Add an agent using an interactive prompt.", "/agent add"
    ),
    CommandSpec(
        "agent",
        "/agent remove <uid> [scratchpad]",
        "Remove an agent and mapping; optionally remove its scratchpad.",
        "/agent remove",
    ),
    CommandSpec("engine", "/engine list", "List configured engines.", "/engine list"),
    CommandSpec(
        "engine", "/engine show <uid>", "Show one configured engine.", "/engine show"
    ),
    CommandSpec(
        "engine",
        "/engine add",
        "Add an engine using an interactive prompt.",
        "/engine add",
    ),
    CommandSpec(
        "engine",
        "/engine remove <uid>",
        "Remove an unassigned engine after confirmation.",
        "/engine remove",
    ),
    CommandSpec(
        "mapping",
        "/mapping list",
        "List persisted agent-to-engine mappings.",
        "/mapping list",
    ),
    CommandSpec(
        "mapping",
        "/mapping show <agent_uid>",
        "Show the engine mapping for one agent.",
        "/mapping show",
    ),
    CommandSpec(
        "mapping",
        "/mapping assign <agent_uid> <engine_uid>",
        "Assign or reassign an engine to an agent.",
        "/mapping assign",
    ),
    CommandSpec(
        "mapping",
        "/mapping unassign <agent_uid>",
        "Remove an agent's engine assignment after confirmation.",
        "/mapping unassign",
    ),
    CommandSpec(
        "files",
        "/files [memory|scratchpad|config]",
        "List managed workspace files, optionally by category.",
        "/files",
    ),
    CommandSpec(
        "files",
        "/file <workspace-relative-path>",
        "View a managed workspace text file.",
        "/file",
    ),
    CommandSpec("files", "/memory list", "List memory files.", "/memory list"),
    CommandSpec(
        "files", "/memory show <filename>", "View one memory file.", "/memory show"
    ),
    CommandSpec(
        "files",
        "/memory append <filename>",
        "Append interactively entered text to a memory file.",
        "/memory append",
    ),
    CommandSpec(
        "files",
        "/memory append-turn <filename>",
        "Append the latest transcript turn to a memory file.",
        "/memory append-turn",
    ),
    CommandSpec(
        "files",
        "/memory clear <filename>",
        "Clear an existing memory file after confirmation.",
        "/memory clear",
    ),
    CommandSpec(
        "files",
        "/memory remove <filename>",
        "Remove an existing memory file after confirmation.",
        "/memory remove",
    ),
    CommandSpec(
        "files", "/scratchpad list", "List agent scratchpads.", "/scratchpad list"
    ),
    CommandSpec(
        "files",
        "/scratchpad show <agent_uid>",
        "View one agent scratchpad.",
        "/scratchpad show",
    ),
    CommandSpec(
        "files",
        "/scratchpad append <agent_uid>",
        "Append interactively entered text to an agent scratchpad.",
        "/scratchpad append",
    ),
    CommandSpec(
        "files",
        "/scratchpad append-turn <agent_uid>",
        "Append the latest transcript turn to an agent scratchpad.",
        "/scratchpad append-turn",
    ),
    CommandSpec(
        "files",
        "/scratchpad clear <agent_uid>",
        "Clear one agent scratchpad after confirmation.",
        "/scratchpad clear",
    ),
    CommandSpec(
        "files",
        "/scratchpad clear-all",
        "Clear every configured agent scratchpad after confirmation.",
        "/scratchpad clear-all",
    ),
    CommandSpec(
        "transcript",
        "/history [metadata]",
        "Show the full chronological transcript, optionally with metadata.",
        "/history",
    ),
    CommandSpec(
        "transcript",
        "/history last <count> [metadata]",
        "Show the last count transcript turns.",
        "/history last",
    ),
    CommandSpec(
        "transcript",
        "/history turns <turn> [<turn> ...] [metadata]",
        "Show selected global transcript turns.",
        "/history turns",
    ),
    CommandSpec(
        "transcript",
        "/history agent <uid> [last <count>] [metadata]",
        "Show transcript turns for one agent.",
        "/history agent",
    ),
    CommandSpec(
        "transcript",
        "/history agents <uid> [<uid> ...] [last <count>] [metadata]",
        "Show transcript turns for selected agents.",
        "/history agents",
    ),
    CommandSpec(
        "transcript",
        "/turn <turn> [metadata]",
        "Show one global transcript turn.",
        "/turn",
    ),
    CommandSpec(
        "transcript",
        "/context [system|team|identity|memory|scratchpad]",
        "Show context from the latest interaction round.",
        "/context",
    ),
    CommandSpec(
        "session",
        "/history-limit",
        "Show the history limit used for subsequent AI turns.",
        "/history-limit",
    ),
    CommandSpec(
        "session",
        "/history-limit <count|all>",
        "Set the history limit for subsequent AI turns; 0 means no history.",
        "/history-limit all",
    ),
    CommandSpec(
        "session",
        "/state",
        "Show current interaction-state counts and round status.",
        "/state",
    ),
    CommandSpec(
        "session",
        "/state clear-provider",
        "Clear all provider continuation state after confirmation.",
        "/state clear-provider",
    ),
    CommandSpec(
        "session",
        "/state clear-history",
        "Clear transcript and provider history after confirmation.",
        "/state clear-history",
    ),
    CommandSpec(
        "session",
        "/state clear-all",
        "Clear the current round and all interaction history after confirmation.",
        "/state clear-all",
    ),
    CommandSpec(
        "session",
        "/provider status",
        "Show provider-history summary without exposing provider state.",
        "/provider status",
    ),
    CommandSpec(
        "session",
        "/provider clear [agent_uid]",
        "Clear all provider state or one agent's current engine chain.",
        "/provider clear",
    ),
)


def resolve_group(name: str) -> CommandGroup | None:
    """Resolve a command group key or alias case-insensitively."""
    normalized = name.strip().lower()

    for group in COMMAND_GROUPS:
        names = {group.key, *(alias.lower() for alias in group.aliases)}
        if normalized in names:
            return group

    return None


def commands_for_group(group_key: str) -> tuple[CommandSpec, ...]:
    """Return commands for a group."""
    return tuple(spec for spec in COMMAND_SPECS if spec.group == group_key)


def static_completions() -> list[str]:
    """Return static completions derived from command definitions."""
    completions = [spec.completion for spec in COMMAND_SPECS]
    completions.append("/exit")

    for group in COMMAND_GROUPS:
        completions.append(f"/help {group.key}")

    return sorted(set(completions))
