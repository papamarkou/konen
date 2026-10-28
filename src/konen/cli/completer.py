from collections.abc import Iterable

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from konen import Orchestrator, Workspace

from .commands import COMMAND_GROUPS, COMMAND_SPECS


class CLICompleter(Completer):
    """Context-aware completion for Konen CLI commands."""

    def __init__(self, workspace: Workspace, orchestrator: Orchestrator) -> None:
        self.workspace = workspace
        self.orchestrator = orchestrator

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Yield completions for the token at the cursor."""
        del complete_event

        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        tokens = text.split()
        if text.endswith(" "):
            context = tokens
            fragment = ""
        else:
            context = tokens[:-1]
            fragment = tokens[-1] if tokens else text

        candidates = self._candidates(context)

        for candidate, metadata in sorted(candidates.items()):
            if candidate.startswith(fragment):
                yield Completion(
                    candidate, start_position=-len(fragment), display_meta=metadata
                )

    def _candidates(self, context: list[str]) -> dict[str, str]:
        """Return static and dynamic next-token candidates."""
        candidates = self._static_candidates(context)
        candidates.update(self._dynamic_candidates(context))
        return candidates

    def _static_candidates(self, context: list[str]) -> dict[str, str]:
        """Derive static candidates from the command registry."""
        candidates: dict[str, str] = {}
        group_titles = {group.key: group.title for group in COMMAND_GROUPS}

        for spec in COMMAND_SPECS:
            tokens = spec.completion.split()
            if len(tokens) <= len(context):
                continue
            if tokens[: len(context)] != context:
                continue

            candidate = tokens[len(context)]
            metadata = (
                group_titles.get(spec.group, spec.group)
                if not context
                else spec.description
            )
            candidates.setdefault(candidate, metadata)

        if not context:
            candidates.setdefault("/exit", "General")

        return candidates

    def _dynamic_candidates(self, context: list[str]) -> dict[str, str]:
        """Return live workspace and transcript candidates."""
        if not context:
            return {}

        agent_uids = self.orchestrator.ai_team.list_agents()
        history_agent_uids = self._history_agent_uids()
        engine_uids = self.orchestrator.llm_pool.list_engines()
        files = self.workspace.list_workspace_files()

        if context == ["/help"]:
            return {
                group.key: f"help group · {group.title}" for group in COMMAND_GROUPS
            }

        if tuple(context) in {
            ("/agent", "show"),
            ("/agent", "select"),
            ("/agent", "remove"),
            ("/mapping", "show"),
            ("/mapping", "unassign"),
            ("/scratchpad", "show"),
            ("/scratchpad", "append"),
            ("/scratchpad", "append-turn"),
            ("/scratchpad", "clear"),
            ("/provider", "clear"),
        }:
            return {agent_uid: "agent" for agent_uid in agent_uids}

        if context == ["/history", "agent"]:
            return {agent_uid: "history agent" for agent_uid in history_agent_uids}

        if len(context) == 3 and context[:2] == ["/agent", "remove"]:
            if context[2] in agent_uids:
                return {"scratchpad": "also delete scratchpad"}

        if tuple(context) in {("/engine", "show"), ("/engine", "remove")}:
            return {engine_uid: "engine" for engine_uid in engine_uids}

        if context == ["/mapping", "assign"]:
            return {agent_uid: "agent" for agent_uid in agent_uids}

        if (
            len(context) == 3
            and context[:2] == ["/mapping", "assign"]
            and context[2] in agent_uids
        ):
            return {engine_uid: "engine" for engine_uid in engine_uids}

        if context == ["/files"]:
            return {
                "memory": "file category",
                "scratchpad": "file category",
                "config": "file category",
            }

        if context == ["/file"]:
            return {
                f"{category}/{filename}": f"{category} file"
                for category, filenames in files.items()
                for filename in filenames
            }

        if tuple(context) in {
            ("/memory", "show"),
            ("/memory", "clear"),
            ("/memory", "remove"),
        }:
            return {filename: "memory file" for filename in files["memory"]}

        if context[:2] == ["/memory", "select"]:
            selected = set(context[2:])
            if selected & {"all", "none"}:
                return {}
            candidates = {
                filename: "memory file"
                for filename in files["memory"]
                if filename not in selected
            }
            if not selected:
                candidates.update(
                    {"all": "all memory files", "none": "no memory files"}
                )
            return candidates

        if tuple(context) in {
            ("/memory", "append"),
            ("/memory", "append-turn"),
        }:
            return {filename: "existing memory file" for filename in files["memory"]}

        if context[:2] == ["/scratchpad", "select-next"]:
            selected = set(context[2:])
            if "@none" in selected:
                return {}
            candidates = {
                agent_uid: "agent scratchpad"
                for agent_uid in agent_uids
                if agent_uid not in selected
            }
            if "@active" not in selected:
                candidates["@active"] = "active agent scratchpad"
            if not selected:
                candidates["@none"] = "no scratchpads"
            return candidates

        if context == ["/context"]:
            return {
                "system": "context section",
                "team": "context section",
                "identity": "context section",
                "memory": "context section",
                "scratchpad": "context section",
            }

        if context == ["/history"]:
            return {
                "metadata": "include metadata",
                "last": "select recent turns",
                "turns": "select global turns",
                "agent": "select one agent",
                "agents": "select multiple agents",
            }

        if context[:2] == ["/history", "turns"]:
            if "metadata" in context:
                return {}
            selected = set(context[2:])
            candidates = {
                str(turn): "transcript turn"
                for turn in self._turns()
                if str(turn) not in selected
            }
            if len(context) > 2:
                candidates["metadata"] = "include metadata"
            return candidates

        if len(context) == 3 and context[:2] == ["/history", "agent"]:
            if context[2] in history_agent_uids:
                return {
                    "last": "limit to recent matching turns",
                    "metadata": "include metadata",
                }

        if context[:2] == ["/history", "agents"]:
            if "metadata" in context or "last" in context:
                return {}
            selected_agents = {
                token for token in context[2:] if token in history_agent_uids
            }
            candidates = {
                agent_uid: "history agent"
                for agent_uid in history_agent_uids
                if agent_uid not in selected_agents
            }
            if selected_agents:
                candidates.update(
                    {
                        "last": "limit to recent matching turns",
                        "metadata": "include metadata",
                    }
                )
            return candidates

        if context == ["/turn"]:
            return {str(turn): "transcript turn" for turn in self._turns()}

        if len(context) == 2 and context[0] == "/turn" and context[1].isdigit():
            return {"metadata": "include metadata"}

        if context == ["/history-limit"]:
            return {
                "all": "use all transcript history",
                "0": "use no transcript history",
            }

        if context == ["/state"]:
            return {
                "clear-provider": "clear provider continuation state",
                "clear-history": "clear transcript and provider history",
                "clear-all": "clear all interaction state",
            }

        if context == ["/provider"]:
            return {
                "status": "show provider-history summary",
                "clear": "clear provider continuation state",
            }

        return {}

    def _history_agent_uids(self) -> list[str]:
        """Return identifiers for configured agents and agents in transcript history."""
        agent_uids = set(self.orchestrator.ai_team.list_agents())
        agent_uids.update(
            round.agent_uid
            for round in self.orchestrator.interaction_state.transcript_history.rounds
        )
        return sorted(agent_uids)

    def _turns(self) -> range:
        """Return available 1-based transcript turns."""
        count = len(self.orchestrator.interaction_state.transcript_history.rounds)
        return range(1, count + 1)
