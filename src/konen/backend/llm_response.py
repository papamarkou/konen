import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LLMResponse:
    """Normalized response from an execution engine."""

    text: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    provider_state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = copy.deepcopy(self.metadata)
        self.provider_state = copy.deepcopy(self.provider_state)
