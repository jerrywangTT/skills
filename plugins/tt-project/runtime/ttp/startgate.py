# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A deferred task's start_when gate as status and the web app show it: its last result (exit code
and a short reason), when it last ran and how long it has said not yet. The daemon records each
verdict (probe_deferred); past `waiting.start_gate_warn_s` still false while the daemon ran (a host
down for hours is no staleness) it is a warning in the feed (state 'status', clears itself once the
gate passes), never a 'needs you' item. A `landed:#<id>` gate whose producer is approved or in a
batch in the push queue names that instead: it is waiting, not stale. Model-free."""
from __future__ import annotations

import json

from . import landed
from .db import deferral

KEY = "start_gates"   # kv: task id -> {when, rc, reason, at, since, run_s} of its start_when's last verdict
WARN_S = 1800         # default of waiting.start_gate_warn_s
GAP_S = 900           # verdicts come every few minutes; a longer gap was the daemon down: only this much counts
_STATES = {"review": "in review", "pushing": "being pushed", "done": "done, not on the branch yet"}


def warn_after(cfg: dict) -> float:
    """waiting.start_gate_warn_s; 0 turns the warning off."""
    return float((cfg.get("waiting") or {}).get("start_gate_warn_s", WARN_S) or 0)


def reason(db, when: str, rc, out: str = "") -> str:
    """Why the gate is still false, in a few words: for `landed:#<id>` (or `ttp landed --task <id>`)
    the producer's state, else the probe's last output line, else what its exit code means."""
    refs = landed.task_refs(when)
    if refs and refs[0].isdigit() and (t := db.task(int(refs[0]))):
        if gone := landed.abandoned(db, t["id"]):
            return f"producer #{t['id']} abandoned ({gone[0][:80]})"
        return f"producer #{t['id']} {_STATES.get(t['status'], t['status'])}"
    last = next((ln.strip() for ln in reversed((out or "").splitlines()) if ln.strip()), "")
    if last:
        return last[:120]
    return {1: "not yet", 75: "busy", 255: "host unreachable"}.get(rc, rc if isinstance(rc, str) else f"exit {rc}")


def record(db, tid: int, when: str, rc, why: str, now: float) -> None:
    """Keeps a not-yet or broken verdict; `since` is when this probe first said so, `run_s` how long
    the daemon has run since then (each gap between verdicts, at most GAP_S)."""
    gates = db.kv(KEY, {}) or {}
    prev = gates.get(str(tid)) or {}
    same = prev.get("when") == when and prev.get("since")
    since = prev.get("since") if same else now
    run_s = float(prev.get("run_s") or 0) + _gap(now - float(prev.get("at") or now)) if same else 0.0
    gates[str(tid)] = {"when": when, "rc": rc, "reason": why, "at": now, "since": since, "run_s": run_s}
    db.set_kv(KEY, gates)


def forget(db, keep: set[int] | None = None, tid: int | None = None) -> None:
    """Drops task `tid`'s record, or every record whose task is not in `keep` (no longer deferred)."""
    gates = db.kv(KEY, {}) or {}
    left = {k: v for k, v in gates.items() if (keep is None or int(k) in keep) and k != str(tid)}
    if left != gates:
        db.set_kv(KEY, left)


def view(db, task: dict, now: float, warn_s: float, gates: dict | None = None) -> dict | None:
    """{text, warn, rc, reason, at, since} of a deferred task's gate, or None before its first verdict
    (or when the probe changed since)."""
    when = deferral(task).get("when")
    g = (db.kv(KEY, {}) if gates is None else gates).get(str(task["id"])) if when else None
    if not g or g.get("when") != when:
        return None
    rc = g.get("rc")
    false_s = now - float(g.get("since") or now)
    run_s = float(g.get("run_s") or 0) + _gap(now - float(g.get("at") or now))   # none recorded: from its next verdict
    warn = bool(warn_s) and run_s >= warn_s
    text = (f"gate {'not yet' if rc in (1, 75, 255) else 'broken'} "
            f"({rc if isinstance(rc, str) else f'exit {rc}'}: {g.get('reason') or '?'}), "
            f"checked {_ago(now - float(g.get('at') or now))} ago, false for {_ago(false_s)}")
    if false_s - run_s >= GAP_S:
        text += f" ({_ago(run_s)} of it with the daemon running)"
    if warn and (queued := push_state(db, when)):
        warn, text = False, f"{text}; {queued}"
    elif warn:
        text += f"; warning: still false after {_ago(warn_s)} of daemon time, check the gate is not stale"
    return {"text": text, "warn": warn, "rc": rc, "reason": g.get("reason"), "at": g.get("at"),
            "since": g.get("since")}


def push_state(db, when: str) -> str | None:
    """'waiting on push batch <id>' (or on the next batch) while a producer the gate waits on, or a
    task carrying its change (its review), has an approved or batched push queue row; else None."""
    ids = set()
    for ref in landed.task_refs(when):
        if ref.isdigit():
            ids |= set(landed.carriers(db, int(ref)))
    if not ids:
        return None
    rows = db.q(f"SELECT status, batch, detail FROM push_queue WHERE status IN ('approved','batched','held') "
                f"AND task IN ({','.join('?' * len(ids))}) ORDER BY id", tuple(sorted(ids)))
    held = next((r for r in rows if r["status"] == "held"), None)
    if held:
        try:
            scope = (json.loads(held.get("detail") or "{}") or {}).get("scope")
        except (TypeError, ValueError, AttributeError):
            scope = None
        return f"held for the token scope {scope}" if scope else "held for a missing token scope"
    batch = next((r["batch"] for r in rows if r["status"] == "batched" and r["batch"]), None)
    return f"waiting on push batch {batch}" if batch else "approved, waiting on the next push batch" if rows else None


def feed(db, warn_s: float, now: float) -> list[dict]:
    """Feed rows (state 'status': not the top section) for each gate false past `warn_s`."""
    gates = db.kv(KEY, {}) or {}
    if not warn_s or not gates:
        return []
    out = []
    for t in db.q("SELECT * FROM tasks WHERE status='queued' AND labels LIKE '%\"start_when:%'"):
        v = view(db, t, now, warn_s, gates)
        if v and v["warn"]:
            out.append({"id": None, "ts": v["since"], "kind": "gate", "severity": "low", "state": "status",
                        "text": f"#{t['id']} {t['title']} waits to start: {v['text']}"})
    return out


def _gap(s: float) -> float:
    return min(max(0.0, s), GAP_S)


def _ago(s: float) -> str:
    s = max(0.0, s)
    return f"{s / 3600:.1f} h" if s >= 5400 else f"{int(s // 60)} min"
