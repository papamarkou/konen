from collections.abc import Iterable

from .engine_factory import EngineFactory
from .llm_engine import LLMEngine
from .workspace import Workspace


class LLMPool:
    """Collection of execution engines indexed by identifier."""

    def __init__(self, engines: Iterable[LLMEngine] = ()) -> None:
        self._engines: dict[str, LLMEngine] = {}
        for engine in engines:
            self._require_engine_absent(engine.uid)
            self._engines[engine.uid] = engine

    @classmethod
    def from_workspace(cls, workspace: Workspace) -> "LLMPool":
        """Build a pool from workspace configuration."""
        engines_data = workspace.load_engines_config()
        engines_config = engines_data["engines"]

        engines: list[LLMEngine] = []
        for engine_uid, config in engines_config.items():
            engines.append(
                EngineFactory.build(
                    model_family=config["model_family"],
                    model_name=config["model_name"],
                    config=config.get("config"),
                    uid=engine_uid,
                )
            )
        return cls(engines)

    def get_engine(self, engine_uid: str) -> LLMEngine:
        """Return the engine with the given identifier."""
        self._require_engine_present(engine_uid)
        return self._engines[engine_uid]

    def list_engines(self) -> list[str]:
        """Return sorted engine identifiers."""
        return sorted(self._engines.keys())

    def add_engine(self, engine: LLMEngine) -> None:
        """Add an engine."""
        self._require_engine_absent(engine.uid)
        self._engines[engine.uid] = engine

    def remove_engine(self, engine_uid: str) -> None:
        """Remove the engine with the given identifier."""
        self._require_engine_present(engine_uid)
        del self._engines[engine_uid]

    def reload_from_workspace(self, workspace: Workspace) -> None:
        """Reload the pool from workspace configuration."""
        pool = self.from_workspace(workspace)
        self._engines = dict(pool._engines)

    def _require_engine_absent(self, engine_uid: str) -> None:
        if engine_uid in self._engines:
            raise ValueError(f"Engine '{engine_uid}' already exists in LLM pool.")

    def _require_engine_present(self, engine_uid: str) -> None:
        if engine_uid not in self._engines:
            raise ValueError(f"Engine '{engine_uid}' does not exist in LLM pool.")
