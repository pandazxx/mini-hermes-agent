"""Delegate a task to `codex` or `claude` (Claude Code) instead of solving it directly.

The agent interprets the task, composes a work prompt for the chosen CLI, and runs
it as a single bash action - see config/extra/delegate.yaml for the system prompt
that enforces this and forbids the agent from doing the work itself.
"""

import os
from pathlib import Path

import typer

from minisweagent import global_config_dir
from minisweagent.agents import get_agent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = builtin_config_dir / "extra" / "delegate.yaml"
DEFAULT_OUTPUT_FILE = global_config_dir / "last_delegate_run.traj.json"
TOOLS = ("codex", "claude")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    task: str = typer.Option(..., "-t", "--task", help="Task/problem statement", prompt=True),
    tool: str = typer.Option(..., "--tool", help=f"CLI to delegate the work to: {' or '.join(TOOLS)}", prompt=True),
    work_model: str = typer.Option(
        "", "--work-model", help="Model id passed to the delegated tool's own -m/--model flag"
    ),
    model_name: str = typer.Option(
        os.getenv("MSWEA_MODEL_NAME"),
        "-m",
        "--model",
        help="Model for the delegating agent itself (defaults to MSWEA_MODEL_NAME env var)",
    ),
    config_spec: list[str] = typer.Option(
        [str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Path to config files, filenames, or key-value pairs"
    ),
    cost_limit: float | None = typer.Option(
        None, "-l", "--cost-limit", help="Cost limit for the delegating agent's own model calls. Set to 0 to disable."
    ),
    output: Path | None = typer.Option(DEFAULT_OUTPUT_FILE, "-o", "--output", help="Output trajectory file"),
) -> dict:
    if tool not in TOOLS:
        raise typer.BadParameter(f"--tool must be one of {TOOLS}, got {tool!r}")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append(
        {
            "agent": {
                "cost_limit": cost_limit if cost_limit is not None else UNSET,
                "output_path": output or UNSET,
            },
            "model": {"model_name": model_name or UNSET},
        }
    )
    config = recursive_merge(*configs)

    model = get_model(config=config.get("model", {}))
    env = get_environment(config.get("environment", {}), default_type="local")
    agent = get_agent(model, env, config.get("agent", {}), default_type="default")
    result = agent.run(task, tool=tool, work_model=work_model)
    print(result.get("submission") or f"(no submission; exit_status={result.get('exit_status')})")
    return result


if __name__ == "__main__":
    app()
