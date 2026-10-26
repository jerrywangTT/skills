# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Who or what started the shutdown that ended the previous boot: best effort, model-free.

The daemon asks once per boot (daemon._note_boot) and keeps the answer in the boot record, so a
question such as "did tt-project reboot this host?" is answered by the record itself. Sources, in
order; each is bounded and never raises:

1. The previous boot's journal (`journalctl -b <boot>`), when this user may read it: a sudo or pkexec
   COMMAND that runs reboot, shutdown, poweroff or halt (or `systemctl reboot|poweroff|halt|kexec`),
   or systemd-logind's 'Power key pressed'; the last one wins. A bare 'System is rebooting/powering
   down' line names no initiator: it is kept only for when the other sources say nothing.
2. A tt-project host-cycle record (`ttp drain --reboot`) from any project on this host that ended that
   boot (hostcycle.planned).
3. An entry in the shared machine-change ledger for this host's machine alias, recorded from
   LEDGER_WINDOW_S before the shutdown (the boot's last heartbeat) up to this boot, that names a power
   cycle or a reboot: the latest one, with its entry id and the project that recorded it.

Else 'unknown (journal not readable)' when the journal could not be read, or 'unknown'. Tests set
TTP_TEST_NO_JOURNAL so the real journalctl never runs; they stub run_journal instead.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from pathlib import Path

JOURNAL_TIMEOUT_S = 5
LEDGER_WINDOW_S = 1800
TEXT_CHARS = 120
_LINE = re.compile(r"\s(?P<ident>[\w.@-]+)(?:\[\d+\])?:\s(?P<msg>.*)$")
_HIDDEN = re.compile(r"insufficient permissions|not seeing messages from other users|no persistent journal"
                     r"|no journal files", re.I)
_STOPS = {"reboot", "shutdown", "poweroff", "halt"}
_SYSTEMCTL_STOPS = {"reboot", "poweroff", "halt", "kexec"}
_LEDGER_WORDS = re.compile(r"power[ _-]?cycl|reboot|power[ _-]?off|shut[ _-]?down", re.I)


def _clip(text, chars: int = TEXT_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= chars else text[:chars - 1] + "…"


def run_journal(boot: str | None) -> tuple[int, str, str] | None:
    """(exit code, stdout, stderr) of journalctl for boot `boot` (the previous boot when it is no
    journal boot id), only the identifiers parse() reads; None when it could not run."""
    if os.environ.get("TTP_TEST_NO_JOURNAL"):
        return None
    hexid = (boot or "").replace("-", "")
    which = hexid if re.fullmatch(r"[0-9a-f]{32}", hexid) else "-1"
    try:
        r = subprocess.run(["journalctl", "-b", which, "--no-pager", "-o", "short", "-n", "5000",
                            "-t", "systemd-logind", "-t", "sudo", "-t", "pkexec"],
                           capture_output=True, text=True, errors="replace", timeout=JOURNAL_TIMEOUT_S,
                           stdin=subprocess.DEVNULL, env={**os.environ, "LC_ALL": "C"})
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return r.returncode, r.stdout or "", r.stderr or ""


def _stop_command(cmd: str) -> str | None:
    """`cmd` shortened to its program and arguments when it stops the host, else None."""
    try:
        words = shlex.split(cmd)
    except ValueError:
        words = cmd.split()
    if not words:
        return None
    prog = Path(words[0]).name
    if prog in _STOPS or (prog == "systemctl" and _SYSTEMCTL_STOPS & set(words[1:])):
        return _clip(" ".join([prog, *words[1:]]), 60)
    return None


def parse(out: str) -> tuple[str | None, str | None]:
    """(initiator, fallback) from journal lines: the last sudo/pkexec stop command or power-key press,
    and a logind 'System is ...' line that names nobody."""
    found = fallback = None
    for line in out.splitlines():
        m = _LINE.search(line)
        if not m:
            continue
        ident, msg = m["ident"], m["msg"].strip()
        if ident == "sudo" and "COMMAND=" in msg \
                and not re.search(r"incorrect password|not allowed|not in sudoers", msg, re.I):
            cmd = _stop_command(msg.split("COMMAND=", 1)[1])
            if cmd:
                found = f"{_clip(msg.split(':', 1)[0], 40)} ran `{cmd}` via sudo (journal)"
        elif ident == "pkexec" and "Executing command" in msg and "[COMMAND=" in msg:
            cmd = _stop_command(msg.split("[COMMAND=", 1)[1].rsplit("]", 1)[0])
            if cmd:
                found = f"{_clip(msg.split(':', 1)[0], 40)} ran `{cmd}` via pkexec (journal)"
        elif ident == "systemd-logind" and "Power key pressed" in msg:
            found = "the power key (journal)"
        elif ident == "systemd-logind" and re.match(r"System is (rebooting|powering down|halting)", msg):
            fallback = f"a request to systemd-logind that names no user (journal: {_clip(msg, 60)})"
    return found, fallback


def _ledger(alias: str | None, start: float, end: float) -> dict | None:
    from . import machine_ledger
    if not alias:
        return None
    hits = []
    for c in machine_ledger.changes([alias], open_only=False):
        # Recorded in the window, or a power-cycle intent planned for a time in it.
        times = [c.get("added")] + ([c.get("at")] if c.get("kind") == machine_ledger.POWER_CYCLE else [])
        inside = [t for t in times if isinstance(t, (int, float)) and start <= t <= end]
        if inside and _LEDGER_WORDS.search(str(c.get("what") or "")):
            hits.append((max(inside), c))
    if not hits:
        return None
    _, c = max(hits, key=lambda h: h[0])
    who = f" by project {_clip(c['by'], 40)}" if c.get("by") else ""
    return {"source": "ledger", "by": str(c.get("by") or ""), "entry": c.get("id"),
            "text": f"machine-change ledger entry #{c.get('id')}{who}: {_clip(c.get('what'), 80)}"}


def find(p, boot: str | None, boot_at: float | None, last_heartbeat: float | None,
         journal=None) -> dict:
    """{source: journal|host-cycle|ledger|unknown, text, ...} for the shutdown that ended boot `boot`;
    `boot_at`: this boot's start; `last_heartbeat`: the daemon's last sign of life on `boot`.
    `journal`: run_journal's answer, read here when None."""
    from . import hostcycle, machines
    readable, fallback = False, None
    try:
        got = run_journal(boot) if journal is None else journal
        if got:
            rc, out, err = got
            readable = rc == 0 and not _HIDDEN.search(f"{out}\n{err}")
            found, fallback = parse(out)
            if found:
                return {"source": "journal", "text": found}
    except Exception:
        pass
    try:
        rec = hostcycle.planned(p, boot, boot_at)
        if rec:
            rid = rec.get("id") or Path(rec["path"]).parent.name
            return {"source": "host-cycle", "project": rec.get("project") or "", "record": rec["path"],
                    "text": f"`ttp drain --reboot` in project {rec.get('project') or '?'} "
                            f"(host-cycle record {rid}: {_clip(rec.get('reason') or 'no reason recorded', 80)})"}
    except Exception:
        pass
    try:
        end = boot_at or time.time()
        here = machines.here()
        hit = _ledger(here[0] if here else None, (last_heartbeat or end) - LEDGER_WINDOW_S, end)
        if hit:
            return hit
    except Exception:
        pass
    if fallback:
        return {"source": "journal", "text": fallback}
    return {"source": "unknown", "text": "unknown" if readable else "unknown (journal not readable)"}
