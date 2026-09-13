from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined, Template

CONFIG_PATH = Path(__file__).parent.parent.parent / "src" / "minisweagent" / "config" / "extra" / "delegate.yaml"


def _render_system_template(tool: str, work_model: str) -> str:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    template = Template(config["agent"]["system_template"], undefined=StrictUndefined)
    return template.render(tool=tool, work_model=work_model)


@pytest.mark.parametrize(
    ("tool", "work_model", "expected_command"),
    [
        ("codex", "", "codex exec --sandbox workspace-write -o /tmp/mswea_delegate_output.txt"),
        ("codex", "gpt-5-codex", "codex exec -m gpt-5-codex --sandbox workspace-write"),
        ("claude", "", "claude -p --permission-mode bypassPermissions --output-format text"),
        ("claude", "sonnet", "claude -p --model sonnet --permission-mode bypassPermissions"),
    ],
)
def test_system_template_renders_tool_specific_delegation_command(tool, work_model, expected_command):
    rendered = _render_system_template(tool, work_model)
    assert expected_command in rendered
    assert rendered.count("codex exec") + rendered.count("claude -p") == 1


def test_system_template_redirects_output_to_a_file_for_both_tools():
    for tool in ("codex", "claude"):
        rendered = _render_system_template(tool, "")
        assert "/tmp/mswea_delegate_output.txt" in rendered
        assert "cat /tmp/mswea_delegate_output.txt" in rendered
    assert "-o /tmp/mswea_delegate_output.txt" in _render_system_template("codex", "")
    assert ">/tmp/mswea_delegate_output.txt" in _render_system_template("claude", "")


def test_system_template_forbids_solving_the_task_directly():
    rendered = _render_system_template("codex", "")
    assert "never solve tasks yourself" in rendered
    assert "EXACTLY THREE allowed bash commands" in rendered
