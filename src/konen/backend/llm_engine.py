import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .llm_response import LLMResponse


@dataclass(slots=True)
class LLMEngine(ABC):
    """Abstract base class for provider execution engines."""

    model_family: str
    model_name: str
    config: dict[str, Any] = field(default_factory=dict)
    uid: str = field(default_factory=lambda: str(uuid4()))
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.config = dict(self.config)
        self._client = self._initialize_client()

    @property
    def client(self) -> Any:
        """Return the provider client."""
        return self._client

    def prepare_provider_state(
        self, states: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Prepare provider state for an engine call."""
        if not states:
            return None
        return copy.deepcopy(states[-1])

    def _build_response_metadata(self) -> dict[str, Any]:
        return {
            "engine_uid": self.uid,
            "model_family": self.model_family,
            "model_name": self.model_name,
        }

    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the provider client from engine configuration."""
        raise NotImplementedError

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        provider_state: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a normalized model response."""
        raise NotImplementedError
