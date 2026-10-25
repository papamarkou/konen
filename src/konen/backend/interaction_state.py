import copy
from dataclasses import dataclass, field
from typing import Any

from .interaction_round import InteractionRound
from .provider_history import ProviderHistory
from .provider_round import ProviderRound
from .transcript_history import TranscriptHistory


@dataclass(slots=True)
class InteractionState:
    """Current interaction round and histories."""

    round: InteractionRound | None = None
    transcript_history: TranscriptHistory = field(default_factory=TranscriptHistory)
    provider_history: ProviderHistory = field(default_factory=ProviderHistory)

    def start_round(
        self,
        agent_uid: str,
        team_context: dict[str, dict[str, str]],
        user_prompt: str | None = None,
        system_context: str = InteractionRound._default_system_context,
        memory_context: dict[str, str] | None = None,
        scratchpad_context: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Start and store an interaction round."""
        if self.round is not None and self.round.transcript.completed_at is None:
            raise RuntimeError(
                "Cannot start a new interaction round while the current "
                "round is incomplete. Clear the current round first."
            )

        self.round = InteractionRound.from_inputs(
            agent_uid=agent_uid,
            team_context=team_context,
            user_prompt=user_prompt,
            system_context=system_context,
            memory_context=memory_context,
            scratchpad_context=scratchpad_context,
            metadata=metadata,
        )

    def append_to_history(self) -> None:
        """Append the completed transcript to transcript history."""
        if self.round is None:
            raise ValueError(
                "InteractionState.round is None. "
                "Start a round before appending to history."
            )

        if self.round.transcript.completed_at is None:
            raise RuntimeError(
                "Cannot append transcript round to history before it is completed."
            )

        self.transcript_history.append(copy.deepcopy(self.round.transcript))

    def append_provider_round(self, round: ProviderRound) -> None:
        """Append provider state for a stored transcript round."""
        if not self.transcript_history.contains(round.transcript_uid):
            raise ValueError(
                "Cannot store provider state for transcript round "
                f"'{round.transcript_uid}' because it is not in history."
            )

        self.provider_history.append(copy.deepcopy(round))

    def clear_round(self) -> None:
        """Clear the current interaction round."""
        self.round = None

    def clear_provider_history(self) -> None:
        """Clear provider continuation state."""
        self.provider_history.clear()

    def clear_history(self) -> None:
        """Clear transcript history and provider state."""
        self.transcript_history.clear()
        self.provider_history.clear()

    def clear(self) -> None:
        """Clear the current round and all history."""
        self.clear_round()
        self.clear_history()
