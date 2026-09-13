"""Stream new lines appended to a file to the console in real time, e.g. while a subprocess is writing to it."""

import threading
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console

console = Console(highlight=False)


def _read_new_text(path: Path, position: int) -> tuple[str, int]:
    if not path.exists():
        return "", position
    text = path.read_text(errors="replace")
    return text[position:], len(text)


@contextmanager
def tail_file_to_console(path: Path, label: str, style: str = "yellow", poll_interval: float = 0.2):
    """Print lines appended to `path` as they appear, until the context exits (flushing any final content)."""
    stop = threading.Event()
    position = 0

    def _poll():
        nonlocal position
        while True:
            new_text, position = _read_new_text(path, position)
            for line in new_text.splitlines():
                console.print(f"[{style}]< {label}[/{style}]: {line}", highlight=False)
            if stop.is_set():
                return
            stop.wait(poll_interval)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=poll_interval * 4)
