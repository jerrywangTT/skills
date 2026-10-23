# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Parallel code tasks that build the same files: a model-free warning, never a refusal.

Two code tasks run side by side and each adds its own copy of one new module: the APIs clash and
a review fails. So when a `code` or `harness` task is added, the repo paths its spec names
(spec_paths: path-like tokens with a known source extension, or whose folder is in the repo) are
compared with (a) the paths other open code tasks' specs name and (b) the files the branches of
other code tasks still to land add or change (`git diff --name-status <base>...<head>`, cached per
head). A branch counts while its task is open, or once done only while it finished within
`overlap.branch_days`, has a head, is not landed (tried.landed_tasks: a push queue landing of it or
of a successor carrying it on, a supersession), not merged into the base nor on the push branch by
patch (landed.py), and its hand-off did not shelve or reject the change. An overlap gives one short
line (warning) in task_add's result and the new task's line in the digest's open tasks, naming the
most recent few tasks (then '+N more'), the shared paths (new files flagged) and the fix:
`depends_on` an open task, `start_when landed:#<id>` a done one. A code worker's prompt lists the
unlanded branches that touch what its spec names (prompt_block). Common files never count: the
`overlap.ignore` globs (changelogs, changesets, manifests, `__init__.py`) and hubs, tracked files
larger than `overlap.hub_bytes`, which most changes touch in passing, and the files the check
commands (`delivery.push_checks`) name, which specs cite as the required suite. Folders are no
paths. A shared file in a hot area, one a review failed on within reviewcap.AREA_WINDOW_S (its main
file) with that change not landed, or a settled area's file, is more than a warning: task_add
chains the new task after the newest open, unlanded task sharing it (chain), unless its spec has a
`parallel: <why>` line. No network, no model.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import time
from pathlib import Path

from . import reviewcap, settled

KINDS = ("code", "harness")
OPEN = ("queued", "running", "review", "pushing", "waiting", "blocked")
LABEL = "overlap:"           # a task's label: overlap:<ids>:<warning>, shown in the digest while queued
PARALLEL_LABEL = "parallel:"  # a task's label: why its spec runs it beside a hot or settled area's task
PARALLEL = re.compile(r"^[ \t]*parallel:[ \t]*(\S.*?)[ \t]*$", re.M)
CACHE = "overlap_diffs.json"  # state/: {head: [[status, path], ...]}, one git diff per head ever
CACHE_MAX = 400
LANDED_CACHE = "overlap_landed.json"   # state/: [head], branch heads found on the push branch by patch
MAX_PATHS = 4                 # shared paths named in one warning
MAX_TASKS = 3                 # other tasks named in one warning, the most recent first
MAX_BRANCHES = 8              # branches a worker's prompt lists, the most recent first
LS_TTL_S = 60
# A slash path without a known extension (bin/tool, skills/tt-x/SKILL): kept when its folder is in the repo.
_PLAIN = re.compile(r"(?<![\w./:@-])((?:[\w.-]+/)+[\w-][\w.-]*)(?![\w/-])")
_ls_cache: dict[str, tuple[float, list[str]]] = {}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, timeout=20,
                          stdin=subprocess.DEVNULL, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def settings(p) -> dict:
    return (p.config().get("overlap") or {}) if p is not None else {}


def tracked(root: Path) -> list[str]:
    """The repo's tracked files (git ls-files), kept LS_TTL_S per root; [] outside git."""
    key, now = str(root), time.time()
    hit = _ls_cache.get(key)
    if hit and now - hit[0] < LS_TTL_S:
        return hit[1]
    try:
        out = _git(root, "ls-files").stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        out = []
    _ls_cache[key] = (now, out)
    return out


def same(a: str, b: str) -> bool:
    """Two repo paths name one file: equal, or one is the other below a folder (a spec may name
    runtime/x.py for plugins/tool/runtime/x.py)."""
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def _dirs(files: list[str]) -> set[str]:
    out = set()
    for f in files:
        parts = f.split("/")[:-1]
        out.update("/".join(parts[i:j]) for i in range(len(parts)) for j in range(i + 1, len(parts) + 1))
    return out


def spec_paths(text: str, root: Path) -> list[str]:
    """The repo paths `text` names: slash paths with a known source extension, or whose folder the
    repo has. Bare file names (named in passing), the repo's folders and URLs are left out."""
    text = re.sub(r"\b[a-z]+://\S+", " ", text or "")
    known = [f for f in settled.files_in(text) if "/" in f]
    plain = [t for t in _PLAIN.findall(text) if t not in known and "." not in t.rsplit("/", 1)[-1]]
    if plain:
        dirs = _dirs(tracked(root))
        plain = [t for t in plain if t.rsplit("/", 1)[0] in dirs and t.rstrip("/") not in dirs]
    return list(dict.fromkeys(f.removeprefix("./") for f in known + plain))


def _ignored(path: str, globs) -> bool:
    return any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(path, "*/" + g) for g in globs or ())


def _hub(path: str, root: Path, hub_bytes: int) -> bool:
    """A tracked file this path names is larger than hub_bytes: most changes touch it in passing."""
    if hub_bytes <= 0:
        return False
    for f in tracked(root):
        if same(f, path):
            try:
                if (root / f).stat().st_size > hub_bytes:
                    return True
            except OSError:
                pass
    return False


def _check_paths(p) -> list[str]:
    """The repo paths the project's check commands (delivery.push_checks) name: specs cite them as
    the required suite, not as files they change."""
    from . import push
    cmds = push.check_list((p.config().get("delivery") or {}).get("push_checks"))
    return spec_paths(" ".join(cmds), p.root) if cmds else []


def keep(p, paths) -> list[str]:
    """`paths` without ignored files, hubs and the files the check commands name."""
    s = settings(p)
    hub = int(s.get("hub_bytes") or 0)
    checks = _check_paths(p)
    return [f for f in paths if not _ignored(f, s.get("ignore")) and not _hub(f, p.root, hub)
            and not any(same(f, c) for c in checks)]


def _cache_path(p) -> Path:
    return p.state / CACHE


def _load_cache(p) -> dict:
    try:
        got = json.loads(_cache_path(p).read_text())
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def diff(p, base: str, head: str, cache: dict) -> list[tuple[str, str]]:
    """[(A|M|D, path)] of `base...head`, from `cache` (keyed by head) or one git diff."""
    if head in cache:
        return [tuple(x) for x in cache[head]]
    r = _git(p.root, "diff", "--name-status", "--no-renames", f"{base}...{head}")
    if r.returncode != 0:
        return []
    out = [(ln[0], ln.split("\t", 1)[1]) for ln in r.stdout.splitlines() if "\t" in ln]
    cache[head] = out
    return out


def _landed_ids(db) -> set[int]:
    """Tasks the records show landed or superseded, themselves or through a successor that carried
    their work on (a review, a fix, a continuation, a rebase): tried.landed_tasks."""
    from .tried import landed_tasks
    return landed_tasks(db)


def _shelved(result) -> bool:
    """The hand-off says its change was shelved or rejected: nothing of it is to land."""
    from .db import load_result
    return str(load_result(result).get("outcome") or "").strip().lower() in ("shelved", "rejected")


def branches(p, skip=()) -> list[dict]:
    """Code tasks' branches still to land, with commits of their own:
    [{id, branch, head, open, files: [(A|M|D, path)]}]. An open task's branch, or a done task's that
    finished (its last run ended) within `overlap.branch_days`, was not shelved or rejected, is not
    landed by the records and not merged into the base; a diff per head is read once and cached in state/."""
    from . import worktree
    days = float(settings(p).get("branch_days") or 14)
    since = time.time() - days * 86400
    rows = p.db.q("SELECT id, branch, status, result, updated FROM tasks WHERE kind IN (%s) AND branch IS NOT NULL "
                  "AND branch != '' AND (status IN (%s) OR (status='done' AND updated>?)) ORDER BY id"
                  % (",".join("?" * len(KINDS)), ",".join("?" * len(OPEN))), (*KINDS, *OPEN, since))
    landed = _landed_ids(p.db)
    rows = [r for r in rows if r["id"] not in landed and r["id"] not in skip
            and (r["status"] in OPEN or not _shelved(r["result"]))]
    done = [r["id"] for r in rows if r["status"] not in OPEN]
    if done:   # `updated` moves with any later write: a done task's finish is its last run's end
        ended = {r["task"]: r["e"] for r in p.db.q("SELECT task, MAX(ended) e FROM runs WHERE task IN (%s) "
                                                     "GROUP BY task" % ",".join("?" * len(done)), done)}
        rows = [r for r in rows if r["status"] in OPEN or (ended.get(r["id"]) or r["updated"] or 0) > since]
    if not rows:
        return []
    try:
        base = worktree.resolve_base(p)
    except (RuntimeError, OSError, subprocess.SubprocessError):
        return []
    refs = [f"refs/heads/{r['branch']}" for r in rows]
    heads = dict(ln.split(" ", 1)[::-1] for ln in _git(p.root, "for-each-ref", "--no-merged", base,
                                                       "--format=%(objectname) %(refname:short)", *refs
                                                       ).stdout.splitlines() if " " in ln)
    cache = _load_cache(p)
    size = len(cache)
    out = []
    for r in rows:
        head = heads.get(r["branch"])
        if head and (files := diff(p, base, head, cache)):
            out.append({"id": r["id"], "branch": r["branch"], "head": head, "open": r["status"] in OPEN,
                        "files": files})
    if len(cache) != size:
        keepers = dict(list(cache.items())[-CACHE_MAX:])
        from .project import durable_write
        try:
            durable_write(_cache_path(p), json.dumps(keepers))
        except OSError:
            pass
    return out


def _on_push_branch(p, b: dict, seen: list) -> bool:
    """A done task's branch is on the push branch under other commits (a rebase, a squash, a replay):
    landed.py's patch checks, local refs only. Read only for branches that touch the new task's
    files; a head found there is kept in `seen` (state/LANDED_CACHE), as it stays there."""
    if b["open"]:
        return False
    if b["head"] in seen:
        return True
    from . import landed
    try:
        hit = landed.on_push_branch(p, b["branch"]) or landed.equivalent(
            p, b["branch"], (p.config().get("delivery") or {}).get("version_bump"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    if hit:
        seen.append(b["head"])
    return bool(hit)


def _load_seen(p) -> list:
    try:
        got = json.loads((p.state / LANDED_CACHE).read_text())
        return got if isinstance(got, list) else []
    except (OSError, ValueError):
        return []


def _save_seen(p, seen: list, size: int) -> None:
    if len(seen) != size:
        from .project import durable_write
        try:
            durable_write(p.state / LANDED_CACHE, json.dumps(seen[-CACHE_MAX:]))
        except OSError:
            pass


def find(p, spec: str, skip=()) -> list[dict]:
    """The other tasks whose spec or unlanded branch names a path `spec` names, the most recent first:
    [{id, open, paths: [(path, new)], branch?, head?}], ignored files and hubs left out."""
    if not settings(p).get("enabled", True):
        return []
    mine = keep(p, spec_paths(spec, p.root))
    if not mine:
        return []
    hits: dict[int, dict] = {}
    open_rows = p.db.q("SELECT id, spec FROM tasks WHERE kind IN (%s) AND status IN (%s) ORDER BY id"
                       % (",".join("?" * len(KINDS)), ",".join("?" * len(OPEN))), (*KINDS, *OPEN))
    for r in open_rows:
        if r["id"] in skip:
            continue
        theirs = spec_paths(r["spec"] or "", p.root)
        shared = [m for m in mine if any(same(m, t) for t in theirs)]
        if shared:
            hits[r["id"]] = {"id": r["id"], "open": True, "paths": {m: False for m in shared}}
    seen = _load_seen(p)
    size = len(seen)
    for b in branches(p, skip):
        shared = {}
        for status, f in b["files"]:
            for m in mine:
                if same(m, f):
                    shared[m] = shared.get(m, False) or status == "A"
        if shared and not _on_push_branch(p, b, seen):
            h = hits.setdefault(b["id"], {"id": b["id"], "open": b["open"], "paths": {}})
            for m, new in shared.items():
                h["paths"][m] = h["paths"].get(m, False) or new
            h.update(branch=b["branch"], head=b["head"])
    _save_seen(p, seen, size)
    return [{**h, "paths": list(h["paths"].items())} for _, h in sorted(hits.items(), reverse=True)]


def advice(hits: list[dict]) -> tuple[list[int], str]:
    """(the ids it names, the fix) for the tasks a warning lists: `depends_on` the open ones, and
    `start_when landed:#<id>` the most recent done one with a head (its landed probe can pass)."""
    shown = hits[:MAX_TASKS]
    deps = [h["id"] for h in shown if h.get("open")]
    land = next((h["id"] for h in shown if not h.get("open") and h.get("head")), None)
    fix = ([f"depends_on [{', '.join(map(str, deps))}]"] if deps else []) + \
        ([f"start_when landed:#{land}"] if land is not None else [])
    return deps + ([land] if land is not None else []), " and ".join(fix)


def warning(hits: list[dict]) -> str:
    """One line naming the most recent few other tasks, the shared paths (new files flagged) and the fix."""
    if not hits:
        return ""
    parts = []
    for h in hits[:MAX_TASKS]:
        names = [f"{f} (new file)" if new else f for f, new in h["paths"][:MAX_PATHS]]
        more = len(h["paths"]) - MAX_PATHS
        parts.append(f"#{h['id']} ({', '.join(names)}{f', +{more} more' if more > 0 else ''})")
    more = len(hits) - MAX_TASKS
    it = "it" if len(hits) == 1 else "them"
    fix = advice(hits)[1]
    return (f"overlaps {'; '.join(parts)}{f'; +{more} more tasks' if more > 0 else ''}: build on {it}, not "
            f"beside {it}{f': {fix}' if fix else ''}")


def label(hits: list[dict], chained: int | None = None) -> str:
    """overlap:<ids the fix names>:<warning>: the digest drops it once the task waits on them all
    (only on `chained` once task_add chained it after that task)."""
    ids = [chained] if chained is not None else advice(hits)[0]
    return f"{LABEL}{','.join(map(str, ids))}:{warning(hits)}" if hits else ""


def digest_note(task: dict) -> str:
    """The overlap warning a queued task carries, '' once it waits on the task(s) it names or has started."""
    if task.get("status") != "queued":
        return ""
    from .db import deferral, dependency_ids
    try:
        labels = json.loads(task.get("labels") or "[]")
    except (ValueError, TypeError):
        return ""
    for lb in labels if isinstance(labels, list) else []:
        if isinstance(lb, str) and lb.startswith(LABEL):
            ids_s, _, text = lb[len(LABEL):].partition(":")
            ids = {int(x) for x in ids_s.split(",") if x.isdigit()}
            when = deferral(task).get("when") or ""
            waits = set(dependency_ids(task)) | {int(x) for x in re.findall(r"#(\d+)", when)}
            from .landed import task_refs
            waits |= {int(x) for x in task_refs(when) if x.isdigit()}
            return "" if ids and ids <= waits else text
    return ""


def hot_files(p, now: float | None = None) -> list[str]:
    """The main files (reviewcap.main_file) of reviews failed within reviewcap.AREA_WINDOW_S whose
    change has not landed: the review areas a new task should not run beside."""
    from .db import dependency_ids
    now = time.time() if now is None else now
    landed, out = _landed_ids(p.db), []
    for rid in reviewcap.failed_reviews(p.db, now - reviewcap.AREA_WINDOW_S):
        r = p.db.task(rid)
        if r and rid not in landed and not set(dependency_ids(r)) & landed and (f := reviewcap.main_file(p, r)):
            out.append(f)
    return out


def waits_on(db, tid: int, ids: set[int], landed: set[int] | None = None, limit: int = 200) -> bool:
    """Whether task `tid` is one of `ids` or waits on one: its depends_on and start_when task refs
    (landed.task_refs), followed through the tasks they name. A landed task's wait is over."""
    from .db import deferral, dependency_ids
    from .landed import task_refs
    landed = _landed_ids(db) if landed is None else landed
    seen, todo = set(), [tid]
    while todo and len(seen) < limit:
        t = db.task(todo.pop())
        if not t or t["id"] in seen or t["id"] in landed:
            continue
        if t["id"] in ids:
            return True
        seen.add(t["id"])
        todo += list(dependency_ids(t)) + [int(r) for r in task_refs(deferral(t).get("when")) if r.isdigit()]
    return False


def chain(p, hits: list[dict], now: float | None = None, lineage: set[int] | None = None) -> tuple[int, str] | None:
    """(id, why) of the newest open, unlanded task in `hits` (find) sharing a settled area's file or a
    hot review area's (hot_files): a new task waits for it to land. None for plain overlaps. A task
    in the new one's `lineage`, or waiting on it (waits_on), is passed over: its work lands only with
    the new task's, so waiting on it would never end."""
    landed = _landed_ids(p.db)
    shown = [h for h in hits if h.get("open") and h["id"] not in landed
             and not (lineage and waits_on(p.db, h["id"], lineage, landed))]
    zones = {settled._base(f) for r in settled.live(p, now) for f in r.get("files") or []} if shown else set()
    hot = None
    for h in shown:   # the most recent first
        paths = [f for f, _ in h["paths"]]
        if f := next((f for f in paths if settled._base(f) in zones), None):
            return h["id"], f"{f} is a settled area"
        hot = hot_files(p, now) if hot is None else hot
        if f := next((f for f in paths if any(same(f, x) for x in hot)), None):
            return h["id"], f"{f} had a failed review within {reviewcap.AREA_WINDOW_S / 3600:g} h"
    return None


def prompt_block(p, task: dict) -> str:
    """A code worker's block: the other unlanded branches that add or change a path its spec names."""
    if (task.get("kind") or "") not in KINDS:
        return ""
    try:
        mine = keep(p, spec_paths(task.get("spec") or "", p.root)) if settings(p).get("enabled", True) else []
        if not mine:
            return ""
        lines, seen = [], _load_seen(p)
        size = len(seen)
        for b in reversed(branches(p, skip={int(task.get("id") or 0)})):
            shared = [(f, s == "A") for s, f in b["files"] if any(same(m, f) for m in mine)]
            if shared and not _on_push_branch(p, b, seen):
                names = ", ".join(f"{f} (new file)" if new else f for f, new in shared[:MAX_PATHS])
                lines.append(f"- #{b['id']} {b['branch']} at {b['head'][:10]}: {names}")
        _save_seen(p, seen, size)
    except Exception:   # a hint, never a reason to fail a dispatch
        return ""
    if not lines:
        return ""
    more = len(lines) - MAX_BRANCHES
    lines = lines[:MAX_BRANCHES] + ([f"- +{more} more"] if more > 0 else [])
    return ("## Unlanded branches that touch your files\n"
            "These other tasks' branches add or change files your spec names and have not landed yet. Build on "
            "or coordinate with them: reuse their module and API (read it with `git show <branch>:<path>`), "
            "never create a second one beside it.\n" + "\n".join(lines))
