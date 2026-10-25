from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class AIAgent:
    """AI agent with a name, description, and identifier."""

    name: str
    description: str
    uid: str = field(default_factory=lambda: str(uuid4()))
