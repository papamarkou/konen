import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from .transcript_round import TranscriptRound


@dataclass(slots=True)
class TranscriptHistory:
    """Chronological transcript history."""

    rounds: list[TranscriptRound] = field(default_factory=list)

    @property
    def first_round(self) -> TranscriptRound | None:
        """Return the earliest transcript round."""
        if not self.rounds:
            return None
        return self.rounds[0]

    @property
    def last_round(self) -> TranscriptRound | None:
        """Return the latest transcript round."""
        if not self.rounds:
            return None
        return self.rounds[-1]

    def append(self, round: TranscriptRound) -> None:
        """Append a transcript round."""
        self.rounds.append(round)

    def clear(self) -> None:
        """Clear transcript history."""
        self.rounds.clear()

    def contains(self, round_uid: str) -> bool:
        """Return whether a transcript round identifier is present."""
        return any(round.uid == round_uid for round in self.rounds)

    def tail(self, history_limit: int | None) -> "TranscriptHistory":
        """Return the requested tail of transcript history."""
        _, rounds = self._select_rounds(history_limit)

        return TranscriptHistory(rounds=[copy.deepcopy(round) for round in rounds])

    def after(self, round_uid: str) -> "TranscriptHistory":
        """Return transcript rounds after the given round."""
        for index, round in enumerate(self.rounds):
            if round.uid == round_uid:
                return TranscriptHistory(
                    rounds=[
                        copy.deepcopy(stored_round)
                        for stored_round in self.rounds[index + 1 :]
                    ]
                )

        raise ValueError(f"Transcript round '{round_uid}' was not found in history.")

    def for_turn(self, turn: int) -> TranscriptRound:
        """Return the transcript round for a 1-based turn."""
        self._validate_turn(turn)
        return copy.deepcopy(self.rounds[turn - 1])

    def for_turns(self, turns: list[int]) -> dict[int, TranscriptRound]:
        """Return transcript rounds for selected 1-based turns."""
        if len(turns) != len(set(turns)):
            raise ValueError("turns must not contain duplicates.")

        selected_rounds: dict[int, TranscriptRound] = {}

        for turn in turns:
            self._validate_turn(turn)
            selected_rounds[turn] = copy.deepcopy(self.rounds[turn - 1])

        return selected_rounds

    def for_agent(self, agent_uid: str) -> "TranscriptHistory":
        """Return transcript history for an agent."""
        return self.for_agents({agent_uid})

    def for_agents(self, agent_uids: set[str]) -> "TranscriptHistory":
        """Return transcript history for selected agents."""
        return TranscriptHistory(
            rounds=[
                copy.deepcopy(round)
                for round in self.rounds
                if round.agent_uid in agent_uids
            ]
        )

    def by_agent(self) -> dict[str, "TranscriptHistory"]:
        """Return transcript history grouped by agent identifier."""
        histories: dict[str, TranscriptHistory] = {}

        for round in self.rounds:
            history = histories.setdefault(round.agent_uid, TranscriptHistory())
            history.append(copy.deepcopy(round))

        return histories

    def to_markdown(
        self,
        fence: str = "~~~",
        history_limit: int | None = None,
        turns: list[int] | None = None,
        include_metadata: bool = False,
    ) -> str:
        """Render transcript history as Markdown."""
        if history_limit is not None and turns is not None:
            raise ValueError("history_limit and turns cannot be used together.")

        if turns is None:
            start_index, rounds = self._select_rounds(history_limit)
            selected_rounds = {
                turn: round for turn, round in enumerate(rounds, start=start_index + 1)
            }
        else:
            selected_rounds = self.for_turns(turns)

        if not selected_rounds:
            return ""

        blocks: list[str] = ["# Transcript history"]

        for turn, round in selected_rounds.items():
            blocks.append(
                round.to_markdown(
                    fence=fence, turn=turn, include_metadata=include_metadata
                ).rstrip()
            )

        return "\n\n".join(blocks).rstrip() + "\n"

    def to_dict(self) -> list[dict[str, object]]:
        """Return transcript history as dictionaries."""
        return [round.to_dict() for round in self.rounds]

    def save_markdown(self, path: str | Path, fence: str = "~~~") -> None:
        """Save transcript history as Markdown."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(fence=fence), encoding="utf-8")

    def save_json(self, path: str | Path) -> None:
        """Save transcript history as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _validate_turn(self, turn: int) -> None:
        """Validate a 1-based transcript turn."""
        if not isinstance(turn, int) or isinstance(turn, bool) or turn <= 0:
            raise ValueError("turn must be a positive integer.")

        if turn > len(self.rounds):
            raise ValueError(
                f"Transcript turn {turn} does not exist. "
                f"History contains {len(self.rounds)} turn(s)."
            )

    def _select_rounds(
        self, history_limit: int | None
    ) -> tuple[int, list[TranscriptRound]]:
        """Select the requested history tail."""
        if history_limit is not None and history_limit < 0:
            raise ValueError("history_limit must be non-negative or None.")

        if history_limit is None:
            start_index = 0
        else:
            start_index = max(len(self.rounds) - history_limit, 0)

        return start_index, self.rounds[start_index:]
