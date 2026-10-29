# Workspaces

A workspace is a directory containing the files for an investigation. Konen uses `config/` for agents, engines, and their assignments, `memory/` for shared notes, and `scratchpad/` for notes for each agent. These files remain available when a session ends.

## Workspace directories

When Konen initializes a new workspace, it creates this structure:

```text
workspace/
├── config/
│   ├── agents.json
│   ├── engines.json
│   └── agent_engine_map.json
├── memory/
└── scratchpad/
```

The three JSON files begin with empty configurations. `memory/` contains no files at first. Konen creates `scratchpad/<agent_uid>.md` when an agent is added.

## Configuration files

The `config/` directory contains three JSON files. `agents.json` defines agents by identifier, name, and description:

```json
{
  "version": "1.0",
  "agents": {
    "researcher": {
      "name": "Researcher",
      "description": "Investigate the question."
    }
  }
}
```

`engines.json` defines engines by identifier, model family, model name, and optional configuration:

```json
{
  "version": "1.0",
  "engines": {
    "gpt_luna": {
      "model_family": "gpt",
      "model_name": "gpt-6-luna"
    }
  }
}
```

`agent_engine_map.json` maps agent identifiers to engine identifiers:

```json
{
  "version": "1.0",
  "agent_engine_map": {
    "researcher": "gpt_luna"
  }
}
```

Here, `researcher` is the agent identifier and `gpt_luna` is the engine identifier. The mapping assigns that engine to that agent. Konen checks that mappings refer to defined agents and engines.

Each file has its own `version` field, reserved for identifying changes to that file's format. Konen writes `"1.0"` when it creates each file, yet the value can be chosen manually. The field is separate from the Konen package version.

## Creating and reloading a workspace
