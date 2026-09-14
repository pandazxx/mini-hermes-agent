"""Agent used by `mini-extra hermes`: an interactive accept/reject loop around one-shot work_llm runs.

The LM delegates via the `delegate_work` tool (the harness builds and runs the actual
`codex`/`claude` command, so the work prompt is recorded verbatim and never has to survive
shell escaping). After each digest submission the user gives a verdict at the REPL:
accept ends the task; reject resets the workspace to the starting commit (`git reset --hard`
+ `git clean -fd`) and feeds the reject reason back into the same LM conversation so the
next round's work prompt can be strengthened. work_llm itself starts every round fresh.

The REPL transcript distinguishes three roles, each with its own color, `>` marking input
sent to that role and `<` marking output received from it: SWE-model (the LM driving this
agent), SWE-agent (this agent's control flow), and work-llm (the delegated tool, tailed
live from its output/progress files).
"""

import shlex
import sys
from pathlib import Path
from typing import NoReturn

from jinja2 import StrictUndefined, Template
from rich.console import Console

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.utils.prompt_user import prompt_session
from minisweagent.exceptions import LimitsExceeded, Submitted, TimeExceeded, UserInterruption
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.utils.tail import tail_file_to_console

console = Console(highlight=False)

_ROLE_STYLES = {"assistant": ("SWE-model", "red", "<"), "exit": ("SWE-agent", "green", "<")}
_DEFAULT_STYLE = ("SWE-agent", "green", ">")

DELEGATE_WORK_TOOL = {
    "type": "function",
    "function": {
        "name": "delegate_work",
        "description": (
            "Delegate a self-contained work prompt to work_llm, which executes it in the repository "
            "and returns its final result message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The complete, self-contained work prompt for work_llm"}
            },
            "required": ["prompt"],
        },
    },
}


class HermesAgentConfig(AgentConfig):
    work_command_templates: dict[str, str] = {}
    """Shell command per work_llm tool name, rendered with pre-shell-quoted `prompt`,
    `output_file` and `progress_file` variables."""
    reject_template: str = ""
    """User message sent to the LM after the user rejects a round."""


class HermesAgent(DefaultAgent):
    def __init__(self, *args, config_class: type = HermesAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.start_commit = ""
        self.accepted_commit = ""
        self.rounds: list[dict] = []

    def run(self, task: str = "", **kwargs) -> dict:
        if self._git("status --porcelain"):
            raise RuntimeError(
                "The repository has uncommitted changes; commit or stash them first "
                "(a rejected round resets the workspace with `git reset --hard` + `git clean -fd`)."
            )
        self.start_commit = self._git("rev-parse HEAD")
        return super().run(task, start_commit=self.start_commit, **kwargs)

    def _git(self, args: str) -> str:
        result = self.env.execute({"command": f"git {args}"})
        if result["returncode"] != 0:
            raise RuntimeError(f"`git {args}` failed: {result['output']}")
        return result["output"].strip()

    def add_messages(self, *messages: dict) -> list[dict]:
        for msg in messages:
            role = msg.get("role") or msg.get("type", "unknown")
            label, color, arrow = _ROLE_STYLES.get(role, _DEFAULT_STYLE)
            console.print(f"[{color}]{arrow} {label}[/{color}]: ", end="")
            console.print(get_content_string(msg), markup=False, highlight=False)
        return super().add_messages(*messages)

    def query(self) -> dict:
        try:
            return super().query()
        except TimeExceeded:
            raise
        except LimitsExceeded:
            if not self._stdin_is_interactive():
                raise
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps, ${self.config.cost_limit}.\n"
                f"Current spend: {self.n_calls} steps, ${self.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            self.config.cost_limit = float(input("New cost limit: "))
            return super().query()

    @staticmethod
    def _stdin_is_interactive() -> bool:
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except (ValueError, OSError):
            return False

    def execute_actions(self, message: dict) -> list[dict]:
        outputs: list[dict] = []
        try:
            for action in message.get("extra", {}).get("actions", []):
                outputs.append(self._execute_action(action))
        except Submitted as e:
            self._handle_verdict(e)
        finally:
            result = self.add_messages(
                *self.model.format_observation_messages(message, outputs, self.get_template_vars())
            )
        return result

    def _execute_action(self, action: dict) -> dict:
        if action.get("name") == "delegate_work":
            return self._delegate_work(action["prompt"])
        return self.env.execute(action)

    def _delegate_work(self, prompt: str) -> dict:
        self.rounds.append({"round": len(self.rounds) + 1, "work_prompt": prompt})
        output_file = Path(self.extra_template_vars.get("work_llm_output_file") or "/tmp/mswea_hermes_output.txt")
        progress_file = Path(self.extra_template_vars.get("work_llm_progress_file") or "/tmp/mswea_hermes_progress.log")
        # Stale content from a previous round must not be re-read or re-tailed
        output_file.unlink(missing_ok=True)
        progress_file.unlink(missing_ok=True)
        command = Template(
            self.config.work_command_templates[self.extra_template_vars["tool"]], undefined=StrictUndefined
        ).render(
            self.get_template_vars()
            | {
                "prompt": shlex.quote(prompt),
                "output_file": shlex.quote(str(output_file)),
                "progress_file": shlex.quote(str(progress_file)),
            }
        )
        console.print("[yellow]> work-llm[/yellow]: ", end="")
        console.print(prompt, markup=False, highlight=False)
        with tail_file_to_console(output_file, "work-llm"), tail_file_to_console(progress_file, "work-llm"):
            result = self.env.execute({"command": command})
        if output_file.exists():
            result = result | {"output": output_file.read_text(errors="replace")}
        self.rounds[-1]["work_llm_output"] = result["output"]
        return result

    def _handle_verdict(self, e: Submitted) -> NoReturn:
        submission = e.messages[0].get("extra", {}).get("submission", "")
        if self.rounds:
            self.rounds[-1]["digest"] = submission
        console.print("[green]< SWE-agent[/green]: ", end="")
        console.print(submission, markup=False, highlight=False)
        console.print(
            "[bold yellow]Accept the work? Press Enter to accept, "
            "or type a reject reason to rework from the starting commit.[/bold yellow]"
        )
        verdict = prompt_session.prompt("> ").strip()
        if verdict.lower() in ("", "y", "yes"):
            self.accepted_commit = self._git("rev-parse HEAD")
            if self.rounds:
                self.rounds[-1]["status"] = "accepted"
            raise e
        if self.rounds:
            self.rounds[-1] |= {"status": "rejected", "reject_reason": verdict}
        self._git(f"reset --hard {self.start_commit}")
        self._git("clean -fd")
        raise UserInterruption(
            {
                "role": "user",
                "content": Template(self.config.reject_template, undefined=StrictUndefined).render(
                    self.get_template_vars() | {"reject_reason": verdict}
                ),
                "extra": {"interrupt_type": "UserRejection", "reject_reason": verdict},
            }
        )

    def serialize(self, *extra_dicts) -> dict:
        hermes = {
            "start_commit": self.start_commit,
            "accepted_commit": self.accepted_commit,
            "rounds": self.rounds,
        }
        return super().serialize({"info": {"hermes": hermes}}, *extra_dicts)
