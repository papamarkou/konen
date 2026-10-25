from collections.abc import Iterable

from .ai_agent import AIAgent
from .workspace import Workspace


class AITeam:
    """Collection of AI agents indexed by identifier."""

    def __init__(self, agents: Iterable[AIAgent] = ()) -> None:
        self._agents: dict[str, AIAgent] = {}
        for agent in agents:
            self._require_agent_absent(agent.uid)
            self._agents[agent.uid] = agent

    @classmethod
    def from_workspace(cls, workspace: Workspace) -> "AITeam":
        """Build a team from workspace configuration."""
        agents_data = workspace.load_agents_config()
        agents_config = agents_data["agents"]

        agents: list[AIAgent] = []
        for agent_uid, config in agents_config.items():
            agents.append(
                AIAgent(
                    name=config["name"],
                    description=config["description"],
                    uid=agent_uid,
                )
            )
        return cls(agents)

    def get_agent(self, agent_uid: str) -> AIAgent:
        """Return the agent with the given identifier."""
        self._require_agent_present(agent_uid)
        return self._agents[agent_uid]

    def list_agents(self) -> list[str]:
        """Return sorted agent identifiers."""
        return sorted(self._agents.keys())

    def to_context(self) -> dict[str, dict[str, str]]:
        """Return agent data as team context."""
        return {
            agent_uid: {"name": agent.name, "description": agent.description}
            for agent_uid, agent in sorted(self._agents.items())
        }

    def add_agent(self, agent: AIAgent) -> None:
        """Add an agent."""
        self._require_agent_absent(agent.uid)
        self._agents[agent.uid] = agent

    def remove_agent(self, agent_uid: str) -> None:
        """Remove the agent with the given identifier."""
        self._require_agent_present(agent_uid)
        del self._agents[agent_uid]

    def reload_from_workspace(self, workspace: Workspace) -> None:
        """Reload the team from workspace configuration."""
        team = self.from_workspace(workspace)
        self._agents = dict(team._agents)

    def _require_agent_absent(self, agent_uid: str) -> None:
        if agent_uid in self._agents:
            raise ValueError(f"Agent '{agent_uid}' already exists in AI team.")

    def _require_agent_present(self, agent_uid: str) -> None:
        if agent_uid not in self._agents:
            raise ValueError(f"Agent '{agent_uid}' does not exist in AI team.")
