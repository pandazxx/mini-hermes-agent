# List available recipes
default:
    @just --list

# Create/update the uv-managed virtualenv with all dev/docs extras
install:
    uv sync --extra full

# Run the test suite (extra args are forwarded to pytest, e.g. `just test -k foo`)
test *args:
    uv run pytest {{ args }}

# Lint the codebase
lint:
    uv run ruff check .

# Format the codebase
fmt:
    uv run ruff format .

# Run lint, format-check and tests without modifying anything (what CI checks)
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest

# Run all configured pre-commit hooks against the whole repo
precommit:
    uv run pre-commit run --all-files

# Run the interactive `mini` agent (args are forwarded, e.g. `just mini -t "fix the bug"`)
mini *args:
    uv run mini {{ args }}

# Run the `hermes` agent, which delegates the task to codex or claude (e.g. `just hermes -t "..." --tool codex`)
hermes *args:
    uv run python -m minisweagent.run.extra.hermes {{ args }}

# Serve the documentation locally
docs:
    uv run mkdocs serve

# Check that the external tools this project relies on are installed
doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    status=0
    for cmd in uv just; do
      if command -v "$cmd" >/dev/null 2>&1; then
        echo "ok      $cmd"
      else
        echo "MISSING $cmd (run 'nix develop' to get it)"
        status=1
      fi
    done
    for cmd in codex claude docker podman; do
      if command -v "$cmd" >/dev/null 2>&1; then
        echo "ok      $cmd"
      else
        echo "note    $cmd not found (optional; only needed by specific environments/run scripts)"
      fi
    done
    exit "$status"
