import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from minisweagent.agents.extra.hermes import HermesAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output

SUBMIT = "cat <<'EOF'\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\nEOF"


def _config(**overrides) -> dict:
    config_path = Path("src/minisweagent/config/extra/hermes.yaml")
    return yaml.safe_load(config_path.read_text())["agent"] | overrides


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", "user.email=t@t.t", "-c", "user.name=t", *args], cwd=tmp_path, check=True)

    git("init", "-q")
    (tmp_path / "tracked.txt").write_text("original\n")
    git("add", "-A")
    git("commit", "-qm", "init")
    return tmp_path


def _tc_model(steps: list[tuple[str, str, dict]]) -> DeterministicToolcallModel:
    outputs = []
    for i, (content, name, args) in enumerate(steps):
        tool_call = {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        outputs.append(
            make_toolcall_output(content, [tool_call], [{"name": name, **args, "tool_call_id": f"call_{i}"}])
        )
    return DeterministicToolcallModel(outputs=outputs)


def _run(repo, model, output_file, verdicts=("",)) -> dict:
    agent = HermesAgent(
        model=model,
        env=LocalEnvironment(cwd=str(repo)),
        **_config(work_command_templates={"codex": "printf 'work_llm progress' > {{ output_file }}"}),
    )
    with patch("minisweagent.agents.utils.prompt_user.prompt_session.prompt", side_effect=list(verdicts)):
        return agent.run("do something", tool="codex", work_model="", work_llm_output_file=str(output_file))


def test_add_messages_prints_swe_model_output_and_swe_agent_input(repo, tmp_path, capsys):
    model = _tc_model(
        [("Delegating.", "delegate_work", {"prompt": "the work prompt"}), ("Digesting.", "bash", {"command": SUBMIT})]
    )
    assert _run(repo, model, tmp_path / "out.txt")["exit_status"] == "Submitted"

    out = capsys.readouterr().out
    assert "< SWE-model" in out
    assert "> SWE-agent" in out
    assert "Delegating." in out
    assert "> work-llm" in out
    assert "the work prompt" in out


def test_delegate_work_tails_the_work_llm_output_file_live(repo, tmp_path, capsys):
    model = _tc_model(
        [("Delegating.", "delegate_work", {"prompt": "a prompt"}), ("Digesting.", "bash", {"command": SUBMIT})]
    )
    _run(repo, model, tmp_path / "out.txt")

    out = capsys.readouterr().out
    assert "< work-llm" in out
    assert "work_llm progress" in out
