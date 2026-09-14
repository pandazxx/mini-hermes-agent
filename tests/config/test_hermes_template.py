from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined, Template

CONFIG_PATH = Path(__file__).parent.parent.parent / "src" / "minisweagent" / "config" / "extra" / "hermes.yaml"


def _agent_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())["agent"]


def _render(template: str, **kwargs) -> str:
    return Template(template, undefined=StrictUndefined).render(**kwargs)


def test_system_template_describes_the_round_protocol_and_forbids_doing_work():
    rendered = _render(_agent_config()["system_template"], tool="claude", start_commit="5432abc")
    assert "never solve tasks yourself" in rendered
    assert "delegate_work" in rendered
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in rendered
    assert "`work_llm` (claude)" in rendered
    assert "5432abc" in rendered
    assert "codex" not in rendered


@pytest.mark.parametrize(
    ("tool", "work_model", "expected_command"),
    [
        ("codex", "", "codex exec --sandbox workspace-write -o '/out.txt' 'the prompt' >'/prog.log' 2>&1"),
        ("codex", "gpt-5-codex", "codex exec -m gpt-5-codex --sandbox workspace-write"),
        (
            "claude",
            "",
            "claude -p --permission-mode bypassPermissions --output-format text 'the prompt' >'/out.txt' 2>&1",
        ),
        ("claude", "sonnet", "claude -p --model sonnet --permission-mode bypassPermissions"),
    ],
)
def test_work_command_templates_render_the_prequoted_prompt_and_paths(tool, work_model, expected_command):
    rendered = _render(
        _agent_config()["work_command_templates"][tool],
        work_model=work_model,
        prompt="'the prompt'",
        output_file="'/out.txt'",
        progress_file="'/prog.log'",
    )
    assert expected_command in rendered


def test_reject_template_renders_the_reason_and_start_commit():
    rendered = _render(_agent_config()["reject_template"], reject_reason="no tests were added", start_commit="5432abc")
    assert "no tests were added" in rendered
    assert "5432abc" in rendered
    assert "delegate_work" in rendered
