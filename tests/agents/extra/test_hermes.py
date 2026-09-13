from pathlib import Path

import yaml

from minisweagent.agents.extra.hermes import HermesAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output


def _config() -> dict:
    config_path = Path("src/minisweagent/config/extra/hermes.yaml")
    return yaml.safe_load(config_path.read_text())["agent"]


def _tc_model(outputs_spec: list[tuple[str, list[dict]]]) -> DeterministicToolcallModel:
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


def test_add_messages_prints_swe_model_output_and_swe_agent_input(capsys):
    model = _tc_model(
        [
            (
                "Delegating.",
                [{"command": "printf done > /tmp/does_not_exist_for_this_test.txt"}],
            ),
            (
                "Digesting.",
                [{"command": "cat <<'EOF'\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\nEOF"}],
            ),
        ]
    )
    agent = HermesAgent(model=model, env=LocalEnvironment(), **_config())
    agent.run("do something", tool="codex", work_model="")

    out = capsys.readouterr().out
    assert "< SWE-model" in out
    assert "> SWE-agent" in out
    assert "Delegating." in out


def test_execute_actions_tails_the_work_llm_output_file_live(tmp_path, capsys):
    output_file = tmp_path / "work_llm_output.txt"
    model = _tc_model(
        [
            (
                "Delegating.",
                [{"command": f"printf 'work_llm progress' > {output_file}"}],
            ),
            (
                "Digesting.",
                [{"command": "cat <<'EOF'\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\nEOF"}],
            ),
        ]
    )
    agent = HermesAgent(model=model, env=LocalEnvironment(), **_config())
    agent.run("do something", tool="codex", work_model="", work_llm_output_file=str(output_file))

    out = capsys.readouterr().out
    assert "< work-llm" in out
    assert "work_llm progress" in out
