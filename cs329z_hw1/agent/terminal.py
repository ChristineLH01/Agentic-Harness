"""Problem 10 (terminal): run one command in the provided sandbox and decide what
the model sees of it.

Observation policy: the model sees the exit code (or that the command timed out
or was rejected), then stdout and stderr. A stream longer than context budget 
is cut to its first and last lines, and the whole stream is saved to a file in 
the workspace, whose path the model is told, so it can read the rest with sed or grep.
"""

from __future__ import annotations

from pathlib import Path

from cs329z_hw1 import sandbox
from cs329z_hw1.tokens import CHARS_PER_TOKEN

OUTPUT_DIR = "outputs"  # where cut streams are saved, inside the workspace
SHARE = 10  # each stream shows at most 1/SHARE of the context budget


class CommandFailed(Exception):
    """A non-zero exit, a timeout or a rejection; the registry makes it a tool_error."""


def run_command(cmd: str, workspace: Path | None, timeout: float, context_budget: int) -> str:
    """Run cmd in the sandbox and return what the model sees. Raise CommandFailed
    with that text when the command did not exit with code 0."""
    result = sandbox.run_terminal(cmd, workspace=workspace, timeout=timeout)
    if result.rejected:
        raise CommandFailed(f"The sandbox rejected the command: {result.reason}")
    
    # The same directory the sandbox ran the command in.
    ws = sandbox.make_workspace(workspace) if workspace is not None else sandbox.default_workspace()
    limit = context_budget // SHARE * CHARS_PER_TOKEN
    status = (f"Timed out: the command ran past the {timeout:g}-second limit and was killed." if result.timed_out
              else f"Exit code: {result.exit_code}")
    parts = [status] + [f"[{name}]\n{_cut(text, name, ws, limit)}"
                        for name, text in (("stdout", result.stdout), ("stderr", result.stderr)) if text]
    shown = "\n".join(parts)
    if result.timed_out or result.exit_code != 0:
        raise CommandFailed(shown)
    return shown


def _cut(text: str, stream: str, ws: Path, limit: int) -> str:
    """text if it fits in limit characters; else its first and last lines, with a
    note that says where the whole stream is saved."""
    if len(text) <= limit:
        return text
    out_dir = ws / OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{stream}-{len(list(out_dir.iterdir())) + 1}.txt"  # a new file per cut stream
    path.write_text(text, encoding="utf-8")
    # Cut at line breaks, so that no shown line is a partial one.
    head, tail = text[: limit // 2], text[-(limit // 2):]
    head = head[: head.rfind("\n") + 1] or head
    tail = tail[tail.find("\n") + 1:] or tail
    rel = path.relative_to(ws)
    lines = text.count("\n") + (not text.endswith("\n"))
    return (f"{head}[... {len(text) - len(head) - len(tail)} characters not shown. The whole {stream} "
            f"({lines} lines) is saved in {rel}; read parts of it with sed -n 'A,Bp' {rel} or grep -n.]\n{tail}")
