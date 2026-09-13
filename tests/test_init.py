"""Tests for minisweagent.__init__."""

import os
import subprocess
import sys


def test_startup_banner_survives_non_utf8_stdout(tmp_path):
    """Importing the package must not crash when stdout can't encode the startup banner (e.g. Windows cp1252)."""
    env = {
        **os.environ,
        "PYTHONIOENCODING": "cp1252",
        "MSWEA_SILENT_STARTUP": "",
        "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path),
    }
    result = subprocess.run([sys.executable, "-c", "import minisweagent"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


def _import_and_print_env(tmp_path, var: str, extra_env: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, "MSWEA_SILENT_STARTUP": "1", "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "global")}
    env.pop(var, None)
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", f"import minisweagent, os; print(os.environ.get({var!r}))"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )


def test_local_env_file_is_loaded_on_import(tmp_path):
    """A project-local .env (e.g. copied from .env.example) is picked up on import, without needing `just`."""
    (tmp_path / ".env").write_text("MSWEA_TEST_DOTENV_PROBE=from-local-dotenv\n")

    result = _import_and_print_env(tmp_path, "MSWEA_TEST_DOTENV_PROBE", {})

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "from-local-dotenv"


def test_local_env_file_does_not_override_the_shell_environment(tmp_path):
    """An already-exported env var must win over the project-local .env (python-dotenv's default behavior)."""
    (tmp_path / ".env").write_text("MSWEA_TEST_DOTENV_PROBE=from-local-dotenv\n")

    result = _import_and_print_env(tmp_path, "MSWEA_TEST_DOTENV_PROBE", {"MSWEA_TEST_DOTENV_PROBE": "from-shell"})

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "from-shell"
