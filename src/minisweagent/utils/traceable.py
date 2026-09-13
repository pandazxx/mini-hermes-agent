"""Per-task traceable output directories: unique task ids and structured on-disk records."""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from minisweagent import __version__

_SECRET_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)


def new_task_id() -> str:
    return uuid.uuid4().hex


def redact_env(env: dict[str, str]) -> dict[str, str]:
    """Replace values of environment variables that look like secrets."""
    return {key: ("<redacted>" if _SECRET_NAME_RE.search(key) else value) for key, value in env.items()}


class TaskTrace:
    """Owns the traceable output directory for a single task, named after its task id."""

    def __init__(self, base_dir: Path, task_id: str = ""):
        self.task_id = task_id or new_task_id()
        self.dir = Path(base_dir) / self.task_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / name

    @property
    def trajectory_path(self) -> Path:
        return self.path("trajectory.traj.json")

    @property
    def task_info_path(self) -> Path:
        return self.path("task.json")

    def write_task_info(self, *, config: dict[str, Any], user_input: dict[str, Any]) -> Path:
        """Record basic task info: versions, config, env vars, user input, and a start timestamp."""
        info = {
            "task_id": self.task_id,
            "mini_version": __version__,
            "created_at": time.time(),
            "user_input": user_input,
            "config": config,
            "env": redact_env(dict(os.environ)),
        }
        self.task_info_path.write_text(json.dumps(info, indent=2, default=str))
        return self.task_info_path

    def record_result(self, result: dict[str, Any]) -> Path:
        """Append the run's result and an end timestamp to the task info file."""
        info = json.loads(self.task_info_path.read_text())
        info["finished_at"] = time.time()
        info["result"] = result
        self.task_info_path.write_text(json.dumps(info, indent=2, default=str))
        return self.task_info_path
