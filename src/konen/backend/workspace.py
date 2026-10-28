import json
import re
from pathlib import Path
from typing import Any, ClassVar

from jsonschema import ValidationError, validate


class Workspace:
    """Manage workspace configuration, memory, and scratchpads."""

    _uid_pattern: ClassVar[str] = r"^[a-z0-9_-]+$"

    def __init__(
        self,
        root_path: Path = Path("."),
        memory_dir_name: str = "memory",
        scratchpad_dir_name: str = "scratchpad",
        config_dir_name: str = "config",
        agents_filename: str = "agents.json",
        engines_filename: str = "engines.json",
        agent_engine_map_filename: str = "agent_engine_map.json",
    ) -> None:
        self.root_path: Path = Path(root_path).resolve()
        self.memory_dir_name: str = memory_dir_name
        self.scratchpad_dir_name: str = scratchpad_dir_name
        self.config_dir_name: str = config_dir_name
        self.agents_filename: str = agents_filename
        self.engines_filename: str = engines_filename
        self.agent_engine_map_filename: str = agent_engine_map_filename

    @property
    def memory_path(self) -> Path:
        """Return the memory directory path."""
        return self.root_path / self.memory_dir_name

    @property
    def scratchpad_path(self) -> Path:
        """Return the scratchpad directory path."""
        return self.root_path / self.scratchpad_dir_name

    @property
    def config_path(self) -> Path:
        """Return the configuration directory path."""
        return self.root_path / self.config_dir_name

    @property
    def agents_path(self) -> Path:
        """Return the agents configuration path."""
        return self.config_path / self.agents_filename

    @property
    def engines_path(self) -> Path:
        """Return the engines configuration path."""
        return self.config_path / self.engines_filename

    @property
    def agent_engine_map_path(self) -> Path:
        """Return the agent-engine mapping path."""
        return self.config_path / self.agent_engine_map_filename

    @property
    def is_initialized(self) -> bool:
        """Return whether required workspace paths exist."""
        return (
            self.root_path.exists()
            and self.memory_path.exists()
            and self.scratchpad_path.exists()
            and self.config_path.exists()
            and self.agents_path.exists()
            and self.engines_path.exists()
            and self.agent_engine_map_path.exists()
        )

    def list_workspace_files(self) -> dict[str, list[str]]:
        """Return workspace files grouped by managed directory."""
        self.ensure_initialized()
        return {
            "memory": self._list_relative_files(self.memory_path),
            "scratchpad": self._list_relative_files(self.scratchpad_path),
            "config": self._list_relative_files(self.config_path),
        }

    def read_workspace_file(self, filename: str) -> str:
        """Read a file from a managed workspace directory."""
        self.ensure_initialized()
        workspace_file = self._resolve_workspace_file_path(filename)

        if not workspace_file.is_file():
            raise FileNotFoundError(f"Workspace file does not exist: '{filename}'.")

        return workspace_file.read_text(encoding="utf-8")

    def load_memory_context(
        self, filenames: tuple[str, ...] | None = None
    ) -> dict[str, str]:
        """Load memory context keyed by workspace-relative path."""
        self.ensure_initialized()

        memory_context: dict[str, str] = {}

        if filenames is not None:
            for filename in filenames:
                path = self._resolve_memory_file_path(filename)
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Memory file does not exist: '{filename}'."
                    )

                workspace_relative_path = path.relative_to(self.root_path).as_posix()
                memory_context[workspace_relative_path] = path.read_text(
                    encoding="utf-8"
                )

            return memory_context

        for path in sorted(self.memory_path.rglob("*")):
            if not path.is_file():
                continue

            workspace_relative_path = path.relative_to(self.root_path).as_posix()

            memory_context[workspace_relative_path] = path.read_text(encoding="utf-8")

        return memory_context

    def load_scratchpad_context(self, agent_uid: str) -> dict[str, str]:
        """Load scratchpad context for an agent."""
        self.ensure_initialized()
        self._require_valid_agent_uid(agent_uid)
        self._require_agent_present(agent_uid)

        scratchpad_file = self.scratchpad_path / f"{agent_uid}.md"

        if not scratchpad_file.is_file():
            raise FileNotFoundError(
                f"Scratchpad does not exist for agent '{agent_uid}'."
            )

        workspace_relative_path = scratchpad_file.relative_to(self.root_path).as_posix()

        return {workspace_relative_path: scratchpad_file.read_text(encoding="utf-8")}

    def _initialize_workspace_directories(self) -> None:
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.scratchpad_path.mkdir(parents=True, exist_ok=True)
        self.config_path.mkdir(parents=True, exist_ok=True)

    def _initialize_workspace_files(self) -> None:
        default_file_contents = (
            (self.agents_path, {"version": "1.0", "agents": {}}),
            (self.engines_path, {"version": "1.0", "engines": {}}),
            (self.agent_engine_map_path, {"version": "1.0", "agent_engine_map": {}}),
        )
        for path, default in default_file_contents:
            if not path.exists():
                with path.open("w", encoding="utf-8") as f:
                    json.dump(default, f, indent=2)
                    f.write("\n")

    def _initialize_scratchpad(self, agent_uid: str) -> None:
        self._require_valid_agent_uid(agent_uid)
        scratchpad_file = self.scratchpad_path / f"{agent_uid}.md"
        scratchpad_file.touch(exist_ok=True)

    def _initialize_scratchpads(self) -> None:
        agents_data = self.load_agents_config()
        agents = agents_data.get("agents", {})
        for agent_uid in agents:
            self._initialize_scratchpad(agent_uid)

    def initialize(self) -> None:
        """Initialize workspace directories, files, and scratchpads."""
        if self.is_initialized:
            return
        self._initialize_workspace_directories()
        self._initialize_workspace_files()
        self._initialize_scratchpads()

    def ensure_initialized(self) -> None:
        """Require an initialized workspace."""
        if not self.is_initialized:
            raise RuntimeError(
                "Workspace is not initialized. Call workspace.initialize() first."
            )

    def clear_scratchpad(self, agent_uid: str) -> None:
        """Clear an agent scratchpad."""
        self.ensure_initialized()
        self._require_valid_agent_uid(agent_uid)
        scratchpad_file = self.scratchpad_path / f"{agent_uid}.md"
        scratchpad_file.touch(exist_ok=True)
        scratchpad_file.write_text("", encoding="utf-8")

    def clear_all_scratchpads(self) -> None:
        """Clear all agent scratchpads."""
        self.ensure_initialized()
        agents_data = self.load_agents_config()
        agents = agents_data.get("agents", {})
        for agent_uid in agents:
            self.clear_scratchpad(agent_uid)

    def append_to_scratchpad(
        self, agent_uid: str, text: str, separator: str = "\n\n"
    ) -> None:
        """Append text to an agent scratchpad with a separator."""
        self.ensure_initialized()
        self._require_valid_agent_uid(agent_uid)
        scratchpad_file = self.scratchpad_path / f"{agent_uid}.md"
        if not scratchpad_file.exists():
            raise FileNotFoundError(
                f"Scratchpad does not exist for agent '{agent_uid}'."
            )
        self._append_text(scratchpad_file, text, separator)

    def clear_memory_file(self, filename: str) -> None:
        """Clear a memory file."""
        self.ensure_initialized()
        memory_file = self._resolve_memory_file_path(filename)

        if not memory_file.is_file():
            raise FileNotFoundError(f"Memory file does not exist: '{filename}'.")

        memory_file.write_text("", encoding="utf-8")

    def append_to_memory(
        self, filename: str, text: str, separator: str = "\n\n"
    ) -> None:
        """Append text to a memory file with a separator."""
        self.ensure_initialized()
        memory_file = self._resolve_memory_file_path(filename)
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        self._append_text(memory_file, text, separator)

    def remove_memory_file(self, filename: str) -> None:
        """Remove a memory file."""
        self.ensure_initialized()
        memory_file = self._resolve_memory_file_path(filename)

        if not memory_file.is_file():
            raise FileNotFoundError(f"Memory file does not exist: '{filename}'.")

        memory_file.unlink()

    def add_agent(self, agent_uid: str, name: str, description: str) -> None:
        """Add an agent and initialize its scratchpad."""
        self.ensure_initialized()
        self._require_valid_agent_uid(agent_uid)
        self._require_agent_absent(agent_uid)

        agents_data = self.load_agents_config()
        agents = agents_data.setdefault("agents", {})
        agents[agent_uid] = {"name": name, "description": description}
        self._save_agents_config(agents_data)
        self._initialize_scratchpad(agent_uid)

    def add_engine(
        self,
        engine_uid: str,
        model_family: str,
        model_name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Add an engine to workspace configuration."""
        self.ensure_initialized()
        self._require_valid_engine_uid(engine_uid)
        self._require_engine_absent(engine_uid)

        engines_data = self.load_engines_config()
        engines = engines_data.setdefault("engines", {})
        engines[engine_uid] = {
            "model_family": model_family,
            "model_name": model_name,
            "config": {} if config is None else config,
        }
        self._save_engines_config(engines_data)

    def assign_engine(self, agent_uid: str, engine_uid: str) -> None:
        """Assign an engine to an agent in workspace configuration."""
        self.ensure_initialized()
        self._require_valid_agent_uid(agent_uid)
        self._require_valid_engine_uid(engine_uid)
        self._require_agent_present(agent_uid)
        self._require_engine_present(engine_uid)

        mapping_data = self.load_agent_engine_map_config()
        mapping = mapping_data.setdefault("agent_engine_map", {})
        mapping[agent_uid] = engine_uid
        self._save_agent_engine_map_config(mapping_data)

    def unassign_engine(self, agent_uid: str) -> None:
        """Remove an agent's engine assignment."""
        self.ensure_initialized()
        self._require_valid_agent_uid(agent_uid)
        self._require_agent_present(agent_uid)

        mapping_data = self.load_agent_engine_map_config()
        mapping = mapping_data.setdefault("agent_engine_map", {})

        if agent_uid not in mapping:
            raise ValueError(f"No engine mapping exists for agent '{agent_uid}'.")

        del mapping[agent_uid]
        self._save_agent_engine_map_config(mapping_data)

    def remove_agent(
        self,
        agent_uid: str,
        remove_scratchpad: bool = False,
    ) -> None:
        """Remove an agent and its engine assignment."""
        self.ensure_initialized()
        self._require_valid_agent_uid(agent_uid)
        self._require_agent_present(agent_uid)

        mapping_data = self.load_agent_engine_map_config()
        mapping = mapping_data.setdefault("agent_engine_map", {})

        if agent_uid in mapping:
            del mapping[agent_uid]
            self._save_agent_engine_map_config(mapping_data)

        agents_data = self.load_agents_config()
        agents = agents_data.setdefault("agents", {})
        del agents[agent_uid]
        self._save_agents_config(agents_data)

        if remove_scratchpad:
            scratchpad_file = self.scratchpad_path / f"{agent_uid}.md"
            if scratchpad_file.exists():
                scratchpad_file.unlink()

    def remove_engine(self, engine_uid: str) -> None:
        """Remove an unassigned engine."""
        self.ensure_initialized()
        self._require_valid_engine_uid(engine_uid)
        self._require_engine_present(engine_uid)

        mapping_data = self.load_agent_engine_map_config()
        mapping = mapping_data.get("agent_engine_map", {})

        assigned_agents = sorted(
            agent_uid
            for agent_uid, mapped_engine_uid in mapping.items()
            if mapped_engine_uid == engine_uid
        )

        if assigned_agents:
            agent_list = ", ".join(assigned_agents)
            raise ValueError(
                f"Cannot remove engine '{engine_uid}' because it is assigned "
                f"to agent(s): {agent_list}. Unassign it first."
            )

        engines_data = self.load_engines_config()
        engines = engines_data.setdefault("engines", {})
        del engines[engine_uid]
        self._save_engines_config(engines_data)

    def list_agents(self) -> list[str]:
        """Return sorted agent identifiers."""
        self.ensure_initialized()
        agents_data = self.load_agents_config()
        return sorted(agents_data.get("agents", {}).keys())

    def list_engines(self) -> list[str]:
        """Return sorted engine identifiers."""
        self.ensure_initialized()
        engines_data = self.load_engines_config()
        return sorted(engines_data.get("engines", {}).keys())

    def load_agents_config(self) -> dict[str, Any]:
        """Load and validate agents configuration."""
        self.ensure_initialized()
        default = {"version": "1.0", "agents": {}}
        agents_data = self._load_json_file(
            self.agents_path,
            label="agents config",
            default=default,
        )
        self._validate_agents_schema(agents_data)
        return agents_data

    def load_engines_config(self) -> dict[str, Any]:
        """Load and validate engines configuration."""
        self.ensure_initialized()
        default = {"version": "1.0", "engines": {}}
        engines_data = self._load_json_file(
            self.engines_path,
            label="engines config",
            default=default,
        )
        self._validate_engines_schema(engines_data)
        return engines_data

    def load_agent_engine_map_config(self) -> dict[str, Any]:
        """Load and validate agent-engine mapping configuration."""
        self.ensure_initialized()
        default = {"version": "1.0", "agent_engine_map": {}}
        agent_engine_map_data = self._load_json_file(
            self.agent_engine_map_path,
            label="agent-engine mapping config",
            default=default,
        )
        self._validate_agent_engine_map_schema(agent_engine_map_data)
        self._validate_agent_engine_map_references(agent_engine_map_data)
        return agent_engine_map_data

    def _append_text(self, path: Path, text: str, separator: str) -> None:
        """Append text with a separator after existing content."""
        prefix = ""

        if text and separator and path.exists():
            size = path.stat().st_size
            if size and separator == "\n\n":
                with path.open("rb") as f:
                    f.seek(max(0, size - 2))
                    tail = f.read()
                trailing_newlines = len(tail) - len(tail.rstrip(b"\n"))
                prefix = "\n" * max(0, 2 - trailing_newlines)
            elif size:
                prefix = separator

        with path.open("a", encoding="utf-8") as f:
            f.write(prefix + text)

    def _save_agents_config(self, agents_data: dict[str, Any]) -> None:
        self._validate_agents_schema(agents_data)
        with self.agents_path.open("w", encoding="utf-8") as f:
            json.dump(agents_data, f, indent=2)
            f.write("\n")

    def _save_engines_config(self, engines_data: dict[str, Any]) -> None:
        self._validate_engines_schema(engines_data)
        with self.engines_path.open("w", encoding="utf-8") as f:
            json.dump(engines_data, f, indent=2)
            f.write("\n")

    def _save_agent_engine_map_config(
        self, agent_engine_map_data: dict[str, Any]
    ) -> None:
        self._validate_agent_engine_map_schema(agent_engine_map_data)
        self._validate_agent_engine_map_references(agent_engine_map_data)
        with self.agent_engine_map_path.open("w", encoding="utf-8") as f:
            json.dump(agent_engine_map_data, f, indent=2)
            f.write("\n")

    def _load_json_file(
        self, path: Path, label: str, default: dict[str, Any]
    ) -> dict[str, Any]:
        """Load JSON, using a default for missing or empty files."""
        if not path.exists():
            return default

        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return default

        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {label}: {path}") from exc
        return data

    def _validate_agents_schema(self, agents_data: dict[str, Any]) -> None:
        """Validate agents configuration structure and fields."""
        schema = {
            "type": "object",
            "required": ["version", "agents"],
            "additionalProperties": False,
            "properties": {
                "version": {"type": "string"},
                "agents": {
                    "type": "object",
                    "propertyNames": {"pattern": self._uid_pattern},
                    "additionalProperties": {
                        "type": "object",
                        "required": ["name", "description"],
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
            },
        }
        self._validate_schema(agents_data, schema, label="agents schema")

    def _validate_engines_schema(self, engines_data: dict[str, Any]) -> None:
        """Validate provider-agnostic engine configuration structure."""
        schema = {
            "type": "object",
            "required": ["version", "engines"],
            "additionalProperties": False,
            "properties": {
                "version": {"type": "string"},
                "engines": {
                    "type": "object",
                    "propertyNames": {"pattern": self._uid_pattern},
                    "additionalProperties": {
                        "type": "object",
                        "required": ["model_family", "model_name"],
                        "additionalProperties": False,
                        "properties": {
                            "model_family": {
                                "type": "string",
                                "enum": ["claude", "gemini", "gpt", "kimi"],
                            },
                            "model_name": {"type": "string"},
                            "config": {"type": "object"},
                        },
                    },
                },
            },
        }
        self._validate_schema(engines_data, schema, label="engines schema")

    def _validate_agent_engine_map_schema(self, mapping_data: dict[str, Any]) -> None:
        """Validate agent-engine mapping structure."""
        schema = {
            "type": "object",
            "required": ["version", "agent_engine_map"],
            "additionalProperties": False,
            "properties": {
                "version": {"type": "string"},
                "agent_engine_map": {
                    "type": "object",
                    "propertyNames": {"pattern": self._uid_pattern},
                    "additionalProperties": {
                        "type": "string",
                        "pattern": self._uid_pattern,
                    },
                },
            },
        }
        self._validate_schema(mapping_data, schema, label="agent-engine-mapping schema")

    def _validate_agent_engine_map_references(
        self, mapping_data: dict[str, Any]
    ) -> None:
        """Validate mapping references against configured agents and engines."""
        agents = self.load_agents_config().get("agents", {})
        engines = self.load_engines_config().get("engines", {})
        mapping = mapping_data.get("agent_engine_map", {})

        for agent_uid, engine_uid in mapping.items():
            if agent_uid not in agents:
                raise ValueError(
                    f"Agent-engine mapping references unknown agent '{agent_uid}'."
                )

            if engine_uid not in engines:
                raise ValueError(
                    f"Agent-engine mapping references unknown engine '{engine_uid}'."
                )

    def _validate_schema(
        self, data: dict[str, Any], schema: dict[str, Any], label: str
    ) -> None:
        """Validate data against a JSON schema."""
        try:
            validate(instance=data, schema=schema)
        except ValidationError as exc:
            location = ".".join(str(part) for part in exc.absolute_path)
            if location:
                raise ValueError(
                    f"Invalid {label} at '{location}': {exc.message}"
                ) from exc
            raise ValueError(f"Invalid {label}: {exc.message}") from exc

    def _resolve_workspace_file_path(self, filename: str) -> Path:
        """Resolve a path within a managed workspace directory."""
        if not filename:
            raise ValueError("filename must not be empty.")

        workspace_file_path = (self.root_path / filename).resolve()
        managed_paths = (
            self.memory_path.resolve(),
            self.scratchpad_path.resolve(),
            self.config_path.resolve(),
        )

        for managed_path in managed_paths:
            try:
                workspace_file_path.relative_to(managed_path)
                return workspace_file_path
            except ValueError:
                continue

        raise ValueError(
            "Invalid workspace filename: path is outside managed workspace directories."
        )

    def _resolve_memory_file_path(self, filename: str) -> Path:
        """Resolve a memory path while rejecting directory traversal."""
        if not filename:
            raise ValueError("filename must not be empty.")

        memory_root_path = self.memory_path.resolve()
        memory_file_path = (memory_root_path / filename).resolve()
        try:
            memory_file_path.relative_to(memory_root_path)
        except ValueError:
            raise ValueError("Invalid memory filename: path escapes memory directory.")

        return memory_file_path

    def _list_relative_files(self, base_path: Path) -> list[str]:
        files = [
            str(path.relative_to(base_path))
            for path in base_path.rglob("*")
            if path.is_file()
        ]
        files.sort()
        return files

    def _require_valid_agent_uid(self, agent_uid: str) -> None:
        if not re.compile(self._uid_pattern).fullmatch(agent_uid):
            raise ValueError("Invalid agent UID. Expected pattern: ^[a-z0-9_-]+$.")

    def _require_valid_engine_uid(self, engine_uid: str) -> None:
        if not re.compile(self._uid_pattern).fullmatch(engine_uid):
            raise ValueError("Invalid engine UID. Expected pattern: ^[a-z0-9_-]+$.")

    def _require_agent_absent(self, agent_uid: str) -> None:
        if agent_uid in self.list_agents():
            raise ValueError(f"Agent '{agent_uid}' already exists.")

    def _require_engine_absent(self, engine_uid: str) -> None:
        if engine_uid in self.list_engines():
            raise ValueError(f"Engine '{engine_uid}' already exists.")

    def _require_agent_present(self, agent_uid: str) -> None:
        if agent_uid not in self.list_agents():
            raise ValueError(f"Agent '{agent_uid}' does not exist.")

    def _require_engine_present(self, engine_uid: str) -> None:
        if engine_uid not in self.list_engines():
            raise ValueError(f"Engine '{engine_uid}' does not exist.")
