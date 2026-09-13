import json
from unittest.mock import patch

import pytest
import typer

from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output
from minisweagent.run.extra.hermes import DEFAULT_CONFIG_FILE, main


def _make_tc_model(outputs_spec: list[tuple[str, list[dict]]]) -> DeterministicToolcallModel:
    """Build a DeterministicToolcallModel from a list of (content, actions) tuples."""
    outputs = []
    for i, (content, actions) in enumerate(outputs_spec):
        tc_actions, tool_calls = [], []
        for j, action in enumerate(actions):
            tool_call_id = f"call_{i}_{j}"
            tc_actions.append({"command": action["command"], "tool_call_id": tool_call_id})
            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "bash", "arguments": f'{{"command": "{action["command"]}"}}'},
                }
            )
        outputs.append(make_toolcall_output(content, tool_calls, tc_actions))
    return DeterministicToolcallModel(outputs=outputs)


def _run_hermes(tool: str, work_model: str, task: str, output) -> dict:
    model = _make_tc_model(
        [
            (
                "Delegating to work_llm.",
                [{"command": "printf 'work_llm did the thing' > /tmp/mswea_hermes_output.txt"}],
            ),
            (
                "Digesting the result.",
                [{"command": "cat <<'EOF'\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nwork_llm did the thing\nEOF"}],
            ),
        ]
    )
    with patch("minisweagent.run.extra.hermes.get_model", return_value=model):
        return main(
            task=task,
            tool=tool,
            work_model=work_model,
            model_name="test-model",
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            cost_limit=10.0,  # each DeterministicToolcallModel step reports cost=1.0; stay well clear of the limit
            output=output,
        )


@pytest.mark.parametrize(("tool"), ["codex", "claude"])
def test_hermes_end_to_end_submits_digest(tmp_path, tool):
    result = _run_hermes(tool, "", "fix the flaky test", tmp_path / "traj.json")
    assert result["exit_status"] == "Submitted"
    assert "work_llm did the thing" in result["submission"]


def test_hermes_renders_the_chosen_tool_into_the_first_messages(tmp_path):
    output = tmp_path / "traj.json"
    _run_hermes("claude", "opus", "do the thing", output)

    messages = json.loads(output.read_text())["messages"]
    system_message, instance_message = messages[0], messages[1]
    assert "claude -p --model opus" in system_message["content"]
    assert "codex" not in system_message["content"]
    assert "work_llm (claude)" in instance_message["content"]
    assert "do the thing" in instance_message["content"]


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
