from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined, Template

CONFIG_PATH = Path(__file__).parent.parent.parent / "src" / "minisweagent" / "config" / "extra" / "hermes.yaml"


def _render_system_template(tool: str, work_model: str) -> str:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    template = Template(config["agent"]["system_template"], undefined=StrictUndefined)
    return template.render(tool=tool, work_model=work_model)


@pytest.mark.parametrize(
    ("tool", "work_model", "expected_command"),
    [
        ("codex", "", 'codex exec --sandbox workspace-write -o "/tmp/mswea_hermes_output.txt"'),
        ("codex", "gpt-5-codex", "codex exec -m gpt-5-codex --sandbox workspace-write"),
        ("claude", "", "claude -p --permission-mode bypassPermissions --output-format text"),
        ("claude", "sonnet", "claude -p --model sonnet --permission-mode bypassPermissions"),
    ],
)
def test_system_template_renders_tool_specific_delegation_command(tool, work_model, expected_command):
    rendered = _render_system_template(tool, work_model)
    assert expected_command in rendered
    assert rendered.count("codex exec") + rendered.count("claude -p") == 1


def test_system_template_redirects_output_to_a_quoted_file_path_for_both_tools():
    for tool in ("codex", "claude"):
        rendered = _render_system_template(tool, "")
        assert '"/tmp/mswea_hermes_output.txt"' in rendered
        assert 'cat "/tmp/mswea_hermes_output.txt"' in rendered
    assert '-o "/tmp/mswea_hermes_output.txt"' in _render_system_template("codex", "")
    assert '>"/tmp/mswea_hermes_output.txt"' in _render_system_template("claude", "")


def test_system_template_passes_the_custom_traceable_paths_through_when_given():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    template = Template(config["agent"]["system_template"], undefined=StrictUndefined)
    rendered = template.render(
        tool="codex",
        work_model="",
        work_llm_output_file="/traceable/task-1/work_llm_output.txt",
        work_llm_progress_file="/traceable/task-1/work_llm_progress.log",
    )
    assert '"/traceable/task-1/work_llm_output.txt"' in rendered
    assert '"/traceable/task-1/work_llm_progress.log"' in rendered
    assert "/tmp/mswea_hermes_output.txt" not in rendered


def test_system_template_passes_the_work_prompt_via_an_unindented_heredoc_not_a_quoted_argument():
    for tool in ("codex", "claude"):
        rendered = _render_system_template(tool, "")
        assert "'<your work prompt>'" not in rendered
        assert "$(cat <<'WORK_PROMPT_EOF'" in rendered
        lines = rendered.splitlines()
        terminator_lines = [line for line in lines if line == "WORK_PROMPT_EOF"]
        assert len(terminator_lines) == 1  # must be flush left (no leading whitespace) to actually terminate


def test_system_template_forbids_solving_the_task_directly():
    rendered = _render_system_template("codex", "")
    assert "never solve tasks yourself" in rendered
    assert "EXACTLY THREE allowed bash commands" in rendered
