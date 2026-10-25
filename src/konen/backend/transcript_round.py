import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class TranscriptRound:
    """Transcript for one user-agent interaction."""

    agent_uid: str
    uid: str = field(default_factory=lambda: str(uuid4()))
    user_prompt: str | None = None
    agent_output: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = copy.deepcopy(self.metadata)

    def start(self) -> None:
        """Mark the transcript round as started."""
        if self.started_at is not None:
            raise RuntimeError("Transcript round has already started.")
        self.started_at = datetime.now(timezone.utc)

    def complete(self) -> None:
        """Mark the transcript round as completed."""
        if self.started_at is None:
            raise RuntimeError("Transcript round cannot complete before it starts.")
        if self.completed_at is not None:
            raise RuntimeError("Transcript round has already completed.")
        self.completed_at = datetime.now(timezone.utc)

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        """Update transcript metadata."""
        self.metadata.update(copy.deepcopy(metadata))

    def to_markdown(
        self,
        fence: str = "~~~",
        turn: int | None = None,
        include_metadata: bool = False,
    ) -> str:
        """Render the transcript round as Markdown."""
        if turn is not None and (
            not isinstance(turn, int) or isinstance(turn, bool) or turn <= 0
        ):
            raise ValueError("turn must be a positive integer or None.")

        heading = "# Transcript round" if turn is None else f"## Turn {turn}"

        lines = [heading, "### Agent", f"Agent identifier: {self.agent_uid}"]

        if include_metadata:
            metadata_text = json.dumps(self.metadata, indent=2, ensure_ascii=False)
            lines.extend(["### Metadata", f"{fence}\n{metadata_text}\n{fence}"])

        lines.extend(
            [
                "### User prompt",
                f"{fence}\n{self.user_prompt or ''}\n{fence}",
                "### Agent output",
                f"{fence}\n{self.agent_output or ''}\n{fence}",
            ]
        )

        return "\n\n".join(lines).rstrip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Return the transcript round as a dictionary."""
        return {
            "agent_uid": self.agent_uid,
            "uid": self.uid,
            "user_prompt": self.user_prompt,
            "agent_output": self.agent_output,
            "started_at": (
                None if self.started_at is None else self.started_at.isoformat()
            ),
            "completed_at": (
                None if self.completed_at is None else self.completed_at.isoformat()
            ),
            "metadata": copy.deepcopy(self.metadata),
        }
