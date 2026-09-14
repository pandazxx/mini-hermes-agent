import json
import subprocess
from unittest.mock import patch

import pytest
import typer

from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output
from minisweagent.run.extra.hermes import DEFAULT_CONFIG_FILE, main

SUBMIT = "cat <<'EOF'\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n{digest}\nEOF"


def _git(repo, *args) -> str:
    cmd = ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", *args]
    return subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("original\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _make_model(steps: list[tuple[str, str, dict]]) -> DeterministicToolcallModel:
    """Build a DeterministicToolcallModel from (content, tool_name, args) tuples."""
    outputs = []
    for i, (content, name, args) in enumerate(steps):
        tool_call = {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        outputs.append(
            make_toolcall_output(content, [tool_call], [{"name": name, **args, "tool_call_id": f"call_{i}"}])
        )
    return DeterministicToolcallModel(outputs=outputs)


def _run_hermes(repo, tmp_path, model, verdicts: list[str], work_command: str) -> dict:
    with (
        patch("minisweagent.run.extra.hermes.get_model", return_value=model),
        patch("minisweagent.agents.utils.prompt_user.prompt_session.prompt", side_effect=verdicts),
    ):
        return main(
            task="Fix bug: xxxx is wrong",
            tool="codex",
            work_model="",
            model_name="test-model",
            config_spec=[
                str(DEFAULT_CONFIG_FILE),
                f"agent.work_command_templates={json.dumps({'codex': work_command})}",
                f"environment.cwd={json.dumps(str(repo))}",
            ],
            cost_limit=100.0,
            output=tmp_path / "traj.json",
            traceable_dir=tmp_path / "traceable",
        )


def test_hermes_accepted_first_round_records_prompt_verbatim(repo, tmp_path):
    prompt = 'fix it: don\'t "break" $(things) `now`\nsecond line'
    model = _make_model(
        [
            ("Delegating.", "delegate_work", {"prompt": prompt}),
            ("Digesting.", "bash", {"command": SUBMIT.format(digest="work_llm did the thing")}),
        ]
    )
    result = _run_hermes(
        repo, tmp_path, model, verdicts=[""], work_command="printf %s {{ prompt }} > {{ output_file }}"
    )
    assert result["exit_status"] == "Submitted"
    assert "work_llm did the thing" in result["submission"]
    assert result["start_commit"] == _git(repo, "rev-parse", "HEAD")
    assert result["accepted_commit"] == result["start_commit"]
    (round_record,) = result["rounds"]
    assert round_record["status"] == "accepted"
    assert round_record["work_prompt"] == prompt
    assert round_record["work_llm_output"] == prompt  # survived shell quoting verbatim
    assert round_record["digest"] == "work_llm did the thing\n"


def test_hermes_rejection_resets_workspace_and_reworks_with_revised_prompt(repo, tmp_path):
    model = _make_model(
        [
            ("Delegating.", "delegate_work", {"prompt": "prompt1"}),
            ("Digesting.", "bash", {"command": SUBMIT.format(digest="digest1")}),
            ("Delegating again.", "delegate_work", {"prompt": "prompt2 with tests"}),
            ("Digesting.", "bash", {"command": SUBMIT.format(digest="digest2")}),
        ]
    )
    result = _run_hermes(
        repo,
        tmp_path,
        model,
        verdicts=["There is no test cases for exceptional scenario", ""],
        work_command="echo attempt >> tracked.txt && touch junk.txt && printf done > {{ output_file }}",
    )
    assert result["exit_status"] == "Submitted"
    assert "digest2" in result["submission"]
    # round 1's changes were rolled back before round 2, so only one attempt survives
    assert (repo / "tracked.txt").read_text() == "original\nattempt\n"
    first, second = result["rounds"]
    assert (first["status"], first["reject_reason"]) == ("rejected", "There is no test cases for exceptional scenario")
    assert (second["status"], second["work_prompt"]) == ("accepted", "prompt2 with tests")

    messages = json.loads((tmp_path / "traj.json").read_text())["messages"]
    reject_messages = [m for m in messages if m.get("extra", {}).get("interrupt_type") == "UserRejection"]
    assert len(reject_messages) == 1
    assert "There is no test cases for exceptional scenario" in reject_messages[0]["content"]
    assert json.loads((tmp_path / "traj.json").read_text())["info"]["hermes"]["rounds"] == result["rounds"]
    (rounds_file,) = (tmp_path / "traceable").glob("*/rounds.json")
    assert json.loads(rounds_file.read_text()) == result["rounds"]


def test_hermes_refuses_to_start_on_a_dirty_worktree(repo, tmp_path):
    (repo / "tracked.txt").write_text("modified\n")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        _run_hermes(repo, tmp_path, _make_model([]), verdicts=[], work_command="true")


def test_hermes_rejects_unknown_tool(tmp_path):
    with pytest.raises(typer.BadParameter):
        main(
            task="do the thing",
            tool="gemini",
            work_model="",
            model_name="test-model",
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            cost_limit=None,
            output=tmp_path / "traj.json",
        )
