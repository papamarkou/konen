from dataclasses import dataclass, field

from .workspace import Workspace


@dataclass(slots=True)
class EngineRouter:
    """Mapping from agent identifiers to engine identifiers."""

    table: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.table = dict(self.table)

    @classmethod
    def from_workspace(cls, workspace: Workspace) -> "EngineRouter":
        """Build a router from workspace configuration."""
        mapping_data = workspace.load_agent_engine_map_config()
        return cls(table=mapping_data["agent_engine_map"])

    def assign_engine(self, agent_uid: str, engine_uid: str) -> None:
        """Assign an engine to an agent."""
        self.table[agent_uid] = engine_uid

    def unassign_engine(self, agent_uid: str) -> None:
        """Remove an agent's engine assignment."""
        if agent_uid not in self.table:
            raise ValueError(f"No engine mapping exists for agent '{agent_uid}'.")
        del self.table[agent_uid]

    def get_engine_uid(self, agent_uid: str) -> str:
        """Return the engine identifier assigned to an agent."""
        try:
            return self.table[agent_uid]
        except KeyError as exc:
            raise ValueError(
                f"No engine mapping exists for agent '{agent_uid}'."
            ) from exc

    def reload_from_workspace(self, workspace: Workspace) -> None:
        """Reload the router from workspace configuration."""
        router = self.from_workspace(workspace)
        self.table = dict(router.table)
