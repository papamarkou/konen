import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from .provider_round import ProviderRound


@dataclass(slots=True)
class ProviderHistory:
    """Provider continuation history."""

    rounds: list[ProviderRound] = field(default_factory=list)

    @property
    def last_round(self) -> ProviderRound | None:
        """Return the latest provider round."""
        if not self.rounds:
            return None
        return self.rounds[-1]

    def append(self, round: ProviderRound) -> None:
        """Append a provider round."""
        if any(
            stored_round.transcript_uid == round.transcript_uid
            for stored_round in self.rounds
        ):
            raise ValueError(
                "Provider state already exists for transcript round "
                f"'{round.transcript_uid}'."
            )

        self.rounds.append(round)

    def for_chain(
        self,
        agent_uid: str,
        engine_uid: str,
        model_family: str,
        model_name: str,
        history_start_uid: str,
    ) -> list[ProviderRound]:
        """Return provider rounds for a continuation chain."""
        return [
            copy.deepcopy(round)
            for round in self.rounds
            if (
                round.agent_uid == agent_uid
                and round.engine_uid == engine_uid
                and round.model_family == model_family
                and round.model_name == model_name
                and round.history_start_uid == history_start_uid
            )
        ]

    def last_for(
        self, agent_uid: str, engine_uid: str, model_family: str, model_name: str
    ) -> ProviderRound | None:
        """Return the latest provider round for an engine selection."""
        for round in reversed(self.rounds):
            if (
                round.agent_uid == agent_uid
                and round.engine_uid == engine_uid
                and round.model_family == model_family
                and round.model_name == model_name
            ):
                return round

        return None

    def clear(self) -> None:
        """Clear provider history."""
        self.rounds.clear()

    def clear_for(self, agent_uid: str, engine_uid: str) -> None:
        """Clear provider history for an agent-engine pair."""
        self.rounds = [
            copy.deepcopy(round)
            for round in self.rounds
            if not (round.agent_uid == agent_uid and round.engine_uid == engine_uid)
        ]

    def to_dict(self) -> list[dict[str, object]]:
        """Return provider history as dictionaries."""
        return [round.to_dict() for round in self.rounds]

    def save_json(self, path: str | Path) -> None:
        """Save provider history as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
