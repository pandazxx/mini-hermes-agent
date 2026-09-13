import json

from minisweagent.utils.traceable import TaskTrace, new_task_id, redact_env


def test_new_task_id_returns_unique_ids():
    assert new_task_id() != new_task_id()


def test_redact_env_masks_secret_looking_keys_but_keeps_others():
    redacted = redact_env({"ANTHROPIC_API_KEY": "sk-secret", "GITHUB_TOKEN": "ghp-secret", "HOME": "/home/user"})
    assert redacted == {"ANTHROPIC_API_KEY": "<redacted>", "GITHUB_TOKEN": "<redacted>", "HOME": "/home/user"}


def test_task_trace_creates_a_subdir_named_after_the_task_id(tmp_path):
    trace = TaskTrace(tmp_path, task_id="abc123")
    assert trace.dir == tmp_path / "abc123"
    assert trace.dir.is_dir()


def test_task_trace_generates_a_task_id_when_none_given(tmp_path):
    trace = TaskTrace(tmp_path)
    assert trace.task_id
    assert trace.dir == tmp_path / trace.task_id


def test_write_task_info_records_versions_config_env_and_user_input(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "should-be-redacted")
    trace = TaskTrace(tmp_path, task_id="t1")
    trace.write_task_info(config={"agent": {"cost_limit": 1.0}}, user_input={"task": "fix the bug"})

    info = json.loads(trace.task_info_path.read_text())
    assert info["task_id"] == "t1"
    assert info["mini_version"]
    assert info["created_at"] > 0
    assert info["user_input"] == {"task": "fix the bug"}
    assert info["config"] == {"agent": {"cost_limit": 1.0}}
    assert info["env"]["OPENAI_API_KEY"] == "<redacted>"


def test_record_result_appends_result_and_finish_timestamp(tmp_path):
    trace = TaskTrace(tmp_path, task_id="t2")
    trace.write_task_info(config={}, user_input={})

    trace.record_result({"exit_status": "Submitted", "submission": "done"})

    info = json.loads(trace.task_info_path.read_text())
    assert info["result"] == {"exit_status": "Submitted", "submission": "done"}
    assert info["finished_at"] >= info["created_at"]
