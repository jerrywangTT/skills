"""`ttp checks --detach` pre-stage: a short command a project names in delivery.check_prestage that
runs in the foreground before the full checks are started in the background. A failure there is
found in the same run (the worker fixes it at once) instead of after the whole detached suite, which
costs a hand-off and a later wake. It only adds an early run: the full checks still run every
command unchanged, and nothing it does marks a check as passed. A timeout, a pre-stage that cannot
be started or a malformed setting never holds the full checks back."""
from __future__ import annotations

import math
import os
import signal
import subprocess
import tempfile
import time

KEY = "check_prestage"
DEFAULT_TIMEOUT_S = 90.0     # under the 120 s a Bash tool call gets by default, so the caller outlives it
MAX_TIMEOUT_S = 900.0
TAIL_LINES = 60
CANNOT_RUN = (126, 127)   # the shell found no such command or could not run it: a setting fault
STOP_ON = (signal.SIGTERM, signal.SIGHUP)   # what ends `ttp checks` (a tool timeout, a run's group kill)


class _Signalled(BaseException):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def parse(v) -> tuple[str, float] | None:
    """(command, timeout) from delivery.check_prestage: a command string, or {"run", "timeout_s"}.
    None when unset or empty; ValueError when malformed."""
    if v is None or v == "" or v == {}:
        return None
    if isinstance(v, str):
        return (v, DEFAULT_TIMEOUT_S) if v.strip() else None
    if not isinstance(v, dict):
        raise ValueError("not a command string or a {\"run\", \"timeout_s\"} object")
    if extra := set(v) - {"run", "timeout_s"}:
        raise ValueError(f"unknown key(s) {', '.join(sorted(map(str, extra)))}")
    run, t = v.get("run"), v.get("timeout_s", DEFAULT_TIMEOUT_S)
    if not isinstance(run, str) or not run.strip():
        raise ValueError("\"run\" is not a non-empty command string")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 < t <= MAX_TIMEOUT_S:
        raise ValueError(f"\"timeout_s\" is not a number of seconds in (0, {MAX_TIMEOUT_S:.0f}]")
    return run, float(t)


def problems(v) -> list[str]:
    try:
        parse(v)
    except ValueError as e:
        return [f"delivery.{KEY}: {e}"]
    return []


def _tail(raw: bytes, n: int = TAIL_LINES) -> str:
    lines = raw.decode(errors="replace").rstrip("\n").splitlines()
    return "\n".join(lines[-n:])


def run(v, cwd: str, env: dict, say) -> int:
    """Run the pre-stage `v` in `cwd`; `say` prints a line. 0: start the full checks (it passed,
    timed out, is unset, malformed or could not be started); 1: it failed, start nothing."""
    try:
        spec = parse(v)
    except ValueError as e:
        say(f"delivery.{KEY} ignored: {e}; starting the full checks without it")
        return 0
    if spec is None:
        return 0
    cmd, timeout = spec
    say(f"pre-stage {cmd!r} (delivery.{KEY}, up to {timeout:.0f} s) before the full checks start in the background")
    t0 = time.monotonic()
    with tempfile.TemporaryFile() as out:
        try:
            proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            say(f"pre-stage not started ({e}); starting the full checks without it")
            return 0
        try:
            rc = _wait(proc, timeout)
        except subprocess.TimeoutExpired:
            rc = None
        if rc is None:
            _stop(proc)
            say(f"pre-stage timed out after {timeout:.0f} s; starting the full checks anyway")
            return 0
        took = time.monotonic() - t0
        if rc == 0:
            say(f"pre-stage passed in {took:.0f} s")
            return 0
        if rc in CANNOT_RUN:
            say(f"pre-stage could not run (exit {rc}): delivery.{KEY} names a command this host lacks; "
                "starting the full checks without it")
            return 0
        out.seek(0)
        tail = _tail(out.read())
    say(f"pre-stage failed (exit {rc}) in {took:.0f} s; the full checks were NOT started. Its output's last "
        f"{TAIL_LINES} lines:\n{tail}\nttp checks: fix this and commit, then run `ttp checks --detach` again "
        "in this run. If the pre-stage itself is broken, not the change, `--no-prestage` starts the full checks "
        "without it")
    return 1


def _wait(proc: subprocess.Popen, timeout: float) -> int:
    """proc.wait(timeout), but a SIGTERM or SIGHUP to this process, or any interruption, first ends
    the pre-stage's group: it runs in a session of its own, so the signal that ends `ttp checks`
    never reaches it. The signal then takes its course as if no handler had been set."""
    old: dict = {}
    def caught(signum, frame):
        raise _Signalled(signum)
    try:
        for sig in STOP_ON:
            if signal.getsignal(sig) is not signal.SIG_IGN:
                old[sig] = signal.signal(sig, caught)
    except ValueError:   # not the main thread: no handlers, the except below still covers it
        pass
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        raise
    except BaseException as e:   # interrupted: never leave its process group running
        _stop(proc)
        if not isinstance(e, _Signalled):
            raise
        prior = old.get(e.signum, signal.SIG_DFL)
        for sig, h in old.items():
            signal.signal(sig, h if h is not None else signal.SIG_DFL)
        old = {}
        if callable(prior):
            prior(e.signum, None)
        else:
            signal.signal(e.signum, signal.SIG_DFL)
            os.kill(os.getpid(), e.signum)
        raise SystemExit(128 + e.signum)
    finally:
        for sig, h in old.items():
            signal.signal(sig, h if h is not None else signal.SIG_DFL)


def _stop(proc: subprocess.Popen) -> None:
    """End the pre-stage's own process group (this command started it in a session of its own)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue
