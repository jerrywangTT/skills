"""A check or schedule command that failed only because a script it ran was rewritten while it ran.

A shell reads a script a piece at a time, so a script edited in place while a shell runs it gives that
shell part of the old file and part of the new: a syntax error, or the wrong commands, that says
nothing about what the command checks. Before a run, watch() records the scripts a command names (a
path in it, a file in the folder it runs in, or a bare command found on PATH, that starts with `#!` or ends in .sh/.bash) and the
other scripts in their folders (the ones they call). rerun() runs the command; a run that failed while
one of them changed is run once more, on the new version, and said as LABEL. The rerun's result counts,
so a real failure still fails; a failure with nothing changed is reported as before.
"""
from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

LABEL = "script changed mid-run, rerun once"
SUFFIXES = (".sh", ".bash")
MAX_FOLDER = 200   # a folder with more entries (/usr/bin) adds only the named script, not its neighbours


def _words(cmd: str) -> list[str]:
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        return list(lex)
    except ValueError:   # an unbalanced quote: the shell will say so, this only watches less
        return cmd.split()


def _is_script(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        if path.suffix in SUFFIXES:
            return True
        with open(path, "rb") as f:
            return f.read(2) == b"#!"
    except OSError:
        return False


def scripts(cmd: str, cwd: str | Path, env: dict | None = None) -> list[Path]:
    """The script files `cmd` (run in `cwd`) names, and the other scripts in their folders."""
    path_var = (env if env is not None else os.environ).get("PATH")
    named: list[Path] = []
    for w in _words(str(cmd)):
        if not w or w.startswith("-") or not (w[0].isalnum() or w[0] in "./~_"):
            continue
        p = Path(os.path.expanduser(w))
        p = p if p.is_absolute() else Path(cwd) / p
        if "/" in w or p.is_file():   # `bash check.sh` reads check.sh from cwd
            named.append(p)
        elif found := shutil.which(w, path=path_var):
            named.append(Path(found))
    out: dict[str, Path] = {}
    for p in named:
        if not _is_script(p):
            continue
        out.setdefault(str(p), p)
        try:
            near = list(p.parent.iterdir())
        except OSError:
            continue
        if len(near) <= MAX_FOLDER:
            for q in near:
                if str(q) not in out and _is_script(q):
                    out[str(q)] = q
    return list(out.values())


def _stat(path: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size


def watch(cmd: str, cwd: str | Path, env: dict | None = None) -> dict[str, tuple | None]:
    """What changed() compares against: each script's identity and content stamp before a run."""
    return {str(p): _stat(str(p)) for p in scripts(cmd, cwd, env)}


def changed(seen: dict[str, tuple | None]) -> list[str]:
    """The watched scripts that were rewritten, replaced or removed since watch()."""
    return [p for p, st in seen.items() if _stat(p) != st]


def note(paths: list[str], cwd: str | Path = "") -> str:
    def short(p: str) -> str:
        try:
            return str(Path(p).relative_to(cwd)) if cwd else p
        except ValueError:
            return p
    return f"{LABEL}: {', '.join(short(p) for p in paths)}"


def rerun(run: Callable[[], T], failed: Callable[[T], bool], cmd: str | None, cwd: str | Path,
          env: dict | None = None, say: Callable[[str], None] | None = None,
          again: Callable[[], bool] | None = None) -> T:
    """run(); when its result failed and a script `cmd` names changed meanwhile, say so and run() once
    more, returning that result. No `cmd` (a check that runs on another host) watches nothing.
    `again` (optional) can veto the rerun, e.g. when no time is left for it."""
    seen = watch(cmd, cwd, env) if cmd else {}
    out = run()
    if not seen or not failed(out):
        return out
    moved = changed(seen)
    if not moved or (again is not None and not again()):
        return out
    if say:
        say(note(moved, cwd))
    return run()
