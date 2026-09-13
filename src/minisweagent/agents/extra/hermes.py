"""Agent used by `mini-extra hermes` to print a real-time, role-colored REPL transcript.

Three roles are distinguished, each with its own color, `>` marking input sent to that role
and `<` marking output received from it:

- SWE-model: the LM driving this agent (assistant messages in/out).
- SWE-agent: this agent's own control flow (actions sent to the environment, observations back).
- work-llm: the delegated `codex`/`claude` tool, tailed live from its output/progress files while
  the delegation command (whose paths are given via the `work_llm_output_file` /
  `work_llm_progress_file` run() kwargs) is executing.
"""

from pathlib import Path

from rich.console import Console

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.utils.tail import tail_file_to_console

console = Console(highlight=False)

_ROLE_STYLES = {"assistant": ("SWE-model", "red", "<"), "exit": ("SWE-agent", "green", "<")}
_DEFAULT_STYLE = ("SWE-agent", "green", ">")


class HermesAgent(DefaultAgent):
    def add_messages(self, *messages: dict) -> list[dict]:
        for msg in messages:
            role = msg.get("role") or msg.get("type", "unknown")
            label, color, arrow = _ROLE_STYLES.get(role, _DEFAULT_STYLE)
            console.print(f"[{color}]{arrow} {label}[/{color}]: ", end="")
            console.print(get_content_string(msg), markup=False, highlight=False)
        return super().add_messages(*messages)

    def execute_actions(self, message: dict) -> list[dict]:
        output_file = Path(self.extra_template_vars.get("work_llm_output_file") or "/dev/null")
        progress_file = Path(self.extra_template_vars.get("work_llm_progress_file") or "/dev/null")
        with tail_file_to_console(output_file, "work-llm"), tail_file_to_console(progress_file, "work-llm"):
            return super().execute_actions(message)
