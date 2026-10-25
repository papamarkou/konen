# Development

Konen uses `uv` to manage its development environment and dependencies. Run the commands below from the repository root.

## Setup

```bash
uv sync
```

This creates the virtual environment in `.venv`. Use `uv run` to run commands in that environment without activating it.

## Tests and code checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

To format the code, run `uv run ruff format .`. To check the 88-character line-length target, run `uv run ruff check src --select E501`.

## Documentation

Build the Sphinx documentation locally:

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to view the result.

## Dependencies

Use `uv add PACKAGE` and `uv remove PACKAGE` for runtime dependencies. Use `uv add --dev PACKAGE` for development tools and `uv add --group docs PACKAGE` for documentation tools. Commit changes to `pyproject.toml` and `uv.lock` together.

## Building

Build the source distribution and wheel with `uv build`. The files are written to `dist/`.
