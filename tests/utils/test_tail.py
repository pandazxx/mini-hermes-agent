import time

from minisweagent.utils.tail import tail_file_to_console


def test_tail_file_to_console_prints_lines_appended_during_the_context(tmp_path, capsys):
    path = tmp_path / "progress.log"
    path.write_text("")

    with tail_file_to_console(path, "work-llm", poll_interval=0.05):
        path.write_text("first line\n")
        time.sleep(0.15)
        path.write_text("first line\nsecond line\n")
        time.sleep(0.15)

    out = capsys.readouterr().out
    assert "work-llm" in out
    assert "first line" in out
    assert "second line" in out


def test_tail_file_to_console_flushes_content_written_right_before_exit(tmp_path, capsys):
    path = tmp_path / "progress.log"

    with tail_file_to_console(path, "work-llm", poll_interval=0.05):
        path.write_text("last minute output\n")

    assert "last minute output" in capsys.readouterr().out


def test_tail_file_to_console_is_a_noop_when_the_file_never_appears(tmp_path, capsys):
    path = tmp_path / "never_created.log"

    with tail_file_to_console(path, "work-llm", poll_interval=0.05):
        time.sleep(0.1)

    assert capsys.readouterr().out == ""
