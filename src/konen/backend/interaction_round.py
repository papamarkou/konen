from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .transcript_history import TranscriptHistory
from .transcript_round import TranscriptRound


@dataclass(slots=True)
class InteractionRound:
    """State for an in-progress user-agent interaction."""

    _default_system_context: ClassVar[str] = (
        "You are part of a team of AI agents coordinated by a human orchestrator. "
        "The human-AI interaction follows a star graph. The human orchestrator is "
        "the central node, and each AI agent is a leaf node. The human orchestrator "
        "prompts exactly one agent per turn. You are the currently active agent. "
        "Respond only as yourself."
    )

    agent_uid: str
    team_context: dict[str, dict[str, str]]
    system_context: str = _default_system_context
    memory_context: dict[str, str] = field(default_factory=dict)
    scratchpad_context: dict[str, str] = field(default_factory=dict)
    transcript: TranscriptRound = field(init=False)

    def __post_init__(self) -> None:
        self._initialize_team_context()
        self.memory_context = dict(self.memory_context)
        self.scratchpad_context = dict(self.scratchpad_context)
        self.transcript = TranscriptRound(agent_uid=self.agent_uid)

    def _initialize_team_context(self) -> None:
        team_context: dict[str, dict[str, str]] = {}
        for agent_uid, agent_data in self.team_context.items():
            team_context[agent_uid] = dict(agent_data)
        self.team_context = team_context
        self._validate_team_context()

    def _validate_team_context(self) -> None:
        if self.agent_uid not in self.team_context:
            raise ValueError(f"Agent '{self.agent_uid}' was not found in team context.")

        for agent_uid, agent_data in self.team_context.items():
            if "name" not in agent_data or "description" not in agent_data:
                raise ValueError(
                    "Agent context must contain 'name' and 'description' "
                    f"for key '{agent_uid}'."
                )

    @property
    def identity_context(self) -> dict[str, str]:
        """Return the active agent's team context."""
        return self.team_context[self.agent_uid]

    @classmethod
    def from_inputs(
        cls,
        agent_uid: str,
        team_context: dict[str, dict[str, str]],
        user_prompt: str | None = None,
        system_context: str = _default_system_context,
        memory_context: dict[str, str] | None = None,
        scratchpad_context: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "InteractionRound":
        """Build a round from orchestration inputs."""
        interaction_round = cls(
            agent_uid=agent_uid,
            team_context=team_context,
            system_context=system_context,
            memory_context={} if memory_context is None else memory_context,
            scratchpad_context={} if scratchpad_context is None else scratchpad_context,
        )

        interaction_round.transcript.user_prompt = user_prompt

        if metadata is not None:
            interaction_round.transcript.update_metadata(metadata)

        if user_prompt is not None:
            interaction_round.transcript.start()

        return interaction_round

    def build_prompt_context(self, fence: str = "~~~") -> str:
        """Build the prompt context as Markdown."""
        lines: list[str] = []

        system_text = self.system_context
        lines.append("## System context")
        lines.append(f"{fence}\n{system_text}\n{fence}")

        team_lines = [
            (
                f"Agent identifier: {agent_uid}\n"
                f"Agent name: {agent_data.get('name', '')}\n\n"
                "Agent role description:\n"
                f"{fence}\n{agent_data.get('description', '')}\n{fence}"
            )
            for agent_uid, agent_data in sorted(self.team_context.items())
        ]
        team_text = "\n\n".join(team_lines)
        lines.append("## Team context")
        lines.append(team_text)

        identity_text = (
            f"Your agent identifier is: {self.agent_uid}\n\n"
            f"Your agent name is: {self.identity_context.get('name', '')}\n\n"
            "Your role description is the following:\n"
            f"{fence}\n{self.identity_context.get('description', '')}\n{fence}"
        )
        lines.append("## Identity context")
        lines.append(identity_text)

        if self.memory_context:
            memory_items = sorted(self.memory_context.items())
            lines.append("## Memory context")
            for filename, contents in memory_items:
                lines.append(f"### {filename}")
                lines.append(f"{fence}\n{contents}\n{fence}")

        if self.scratchpad_context:
            scratchpad_items = sorted(self.scratchpad_context.items())
            lines.append("## Scratchpad context")
            for filename, contents in scratchpad_items:
                lines.append(f"### {filename}")
                lines.append(f"{fence}\n{contents}\n{fence}")

        return "\n\n".join(lines).rstrip() + "\n"

    def build_prompt_input(
        self,
        history: TranscriptHistory,
        history_limit: int | None = None,
        fence: str = "~~~",
    ) -> str:
        """Build transcript history and the current user prompt as Markdown."""
        if self.transcript.user_prompt is None:
            raise ValueError(
                "Cannot build prompt input when transcript.user_prompt is None."
            )

        blocks: list[str] = []

        history_text = history.to_markdown(fence=fence, history_limit=history_limit)

        if history_text:
            blocks.append(history_text.rstrip())

        blocks.extend(
            [
                "# User prompt",
                f"{fence}\n{self.transcript.user_prompt}\n{fence}",
            ]
        )

        return "\n\n".join(blocks).rstrip() + "\n"

    def build_prompt(
        self,
        history: TranscriptHistory,
        history_limit: int | None = None,
        fence: str = "~~~",
    ) -> str:
        """Build the complete prompt as Markdown."""
        blocks: list[str] = [
            "# Prompt context",
            self.build_prompt_context(fence=fence).rstrip(),
            self.build_prompt_input(
                history=history, history_limit=history_limit, fence=fence
            ).rstrip(),
        ]

        return "\n\n".join(blocks).rstrip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Return the round as a dictionary."""
        return {
            "agent_uid": self.agent_uid,
            "team_context": {
                agent_uid: dict(agent_data)
                for agent_uid, agent_data in self.team_context.items()
            },
            "system_context": self.system_context,
            "memory_context": dict(self.memory_context),
            "scratchpad_context": dict(self.scratchpad_context),
            "transcript": self.transcript.to_dict(),
        }

    def save_prompt_context(self, path: str | Path, fence: str = "~~~") -> None:
        """Save the prompt context as Markdown."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.build_prompt_context(fence=fence), encoding="utf-8")

    def save_prompt(
        self,
        path: str | Path,
        history: TranscriptHistory,
        history_limit: int | None = None,
        fence: str = "~~~",
    ) -> None:
        """Save the complete prompt as Markdown."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self.build_prompt(
                history=history, history_limit=history_limit, fence=fence
            ),
            encoding="utf-8",
        )
