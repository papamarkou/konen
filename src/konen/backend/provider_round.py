import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProviderRound:
    """Provider continuation state for a transcript round."""

    transcript_uid: str
    agent_uid: str
    engine_uid: str
    model_family: str
    model_name: str
    history_start_uid: str
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = copy.deepcopy(self.state)

    def to_dict(self) -> dict[str, Any]:
        """Return the provider round as a dictionary."""
        return {
            "transcript_uid": self.transcript_uid,
            "agent_uid": self.agent_uid,
            "engine_uid": self.engine_uid,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "history_start_uid": self.history_start_uid,
            "state": copy.deepcopy(self.state),
        }
