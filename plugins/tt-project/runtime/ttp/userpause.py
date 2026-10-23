# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Holds only the user can lift, and holds only time lifts.

A user pause is a probe that only tests for a marker file the user's word creates (by default one named
`user-*`, e.g. `test -e state/user-resume-tuning || exit 1`; coordinator.user_pause_marker, a regex
searched in the tested path, changes it): file tests of user markers joined by `||` or `&&`, or an `||`
of a user marker with other plain file tests (the user's word still ends it), optionally ending in
`|| exit 1`. A marker AND-ed or OR-ed with anything else (ssh, a command, a pipe) is a live probe. No
model turn can make a user pause pass, and the daemon already wakes or requeues the task once it does.
is_probe is the one classifier: the blocker sweep carries such a hold without an outcome, the waiting
check leaves it out of wait_stale and the progress check raises no probe stall for it. A time-only deferral (a `start_after` with no `start_when`)
also ends on its own; the sweep carries it the same way. Both stay listed for the user.
"""
from __future__ import annotations

import json
import re
import time

from . import anchors
from .db import deferral, load_result

DEFAULT_MARKER = r"(?:^|/)user-[\w.-]+$"
_PATH = r"(?:[^\s;|&`()<>$]|\$(?=[\w{]))+"   # a plain path: a variable may expand, nothing runs
_TERM = re.compile(rf"\s*(?:test\s+-[ef]\s+(?P<p1>{_PATH})|\[\s+-[ef]\s+(?P<p2>{_PATH})\s+\])\s*")
_OP = re.compile(r"(\|\||&&)")
_TAIL = re.compile(r"\s*\|\|\s*exit\s+1\s*;?\s*$|\s*;?\s*$")


def settings(where) -> dict:
    """Project settings from `where`: a settings dict, a Project, or anything else (no settings)."""
    if isinstance(where, dict):
        return where
    cfg = getattr(where, "config", None)
    try:
        return cfg() if callable(cfg) else {}
    except Exception:  # an unreadable project.json must not break a sweep
        return {}


def marker(cfg: dict | None) -> re.Pattern:
    """The user-pause marker pattern: coordinator.user_pause_marker, or the default when unset or invalid."""
    raw = ((cfg or {}).get("coordinator") or {}).get("user_pause_marker")
    if isinstance(raw, str) and raw.strip():
        try:
            return re.compile(raw)
        except re.error:
            pass
    return re.compile(DEFAULT_MARKER)


def is_probe(probe: object, cfg: dict | None = None) -> bool:
    """Whether `probe` is a user pause: plain file tests only, every one a user marker (any of `||`
    and `&&` between them), or an `||` chain with at least one user marker, optionally ending in
    `|| exit 1`."""
    text = _TAIL.sub("", str(probe or ""), count=1)
    parts = _OP.split(text)
    terms = [_TERM.fullmatch(t) for t in parts[0::2]]
    if not text.strip() or not all(terms):
        return False
    rx = marker(cfg)
    user = [bool(rx.search((m.group("p1") or m.group("p2")).strip("'\""))) for m in terms]
    return all(user) or (any(user) and set(parts[1::2]) <= {"||"})


def probe_of(task: dict, cfg: dict | None = None) -> str | None:
    """The user-pause probe that is `task`'s only hold, or None. Blocked: its `when:` anchor. Waiting
    (or queued after a waiting hand-off, unless only `start_after` defers it now): its retry_when.
    Deferred: its start_when."""
    if task.get("status") == "blocked":
        a = anchors.anchor(task)
        return a[1] if a and a[0] == "when" and is_probe(a[1], cfg) else None
    if task.get("status") not in ("queued", "waiting"):
        return None
    when = deferral(task).get("when")
    if when:
        return when if is_probe(when, cfg) else None
    if time_only(task):   # deferred by start_after alone: an earlier wait's probe no longer holds it
        return None
    prev = load_result(task.get("result"))
    rw = prev.get("retry_when") if prev.get("status") == "waiting" else None
    return rw if is_probe(rw, cfg) else None


def time_only(task: dict, now: float | None = None) -> bool:
    """A queued task deferred only by `start_after`, still ahead."""
    now = time.time() if now is None else now
    if task.get("status") != "queued":
        return False
    try:
        labels = json.loads(task.get("labels") or "[]")
    except (TypeError, ValueError):
        return False
    return (isinstance(labels, list) and any(isinstance(lb, str) and lb.startswith("start_after:") for lb in labels)
            and not deferral(task).get("when") and float(task.get("not_before") or 0) > now)
