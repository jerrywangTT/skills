# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Project tools still enforcing a Restrictions item that was loosened or retired.

After the user's word loosens or removes a Restrictions item (a quoted edit, or a `replaces` of a
whole Restrictions section with looser wording), or `over` retires one, the project's own tools
(bin/, hooks/, schedules) may still carry the old rule. restriction_tools names them in a queued
`restriction_tools` event for the next digest; it never edits anything. coordinator.apply() calls it
once per turn, with the turn's tasks, so a tool a task of that batch already names is left out.

only_blocks_lines gives the daily review, per Restrictions item, the 7-day count of blocked runs and
asks citing it against refused actions citing it, and lists an item that only blocked work: evidence
for the coordinator's one-time ask to loosen it. Only the user's own words loosen a rule.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .project import Project

# A term a tool would carry word for word: `code`, a path or dotted name, an UPPER_CASE name.
_TOOL_TERM_RE = re.compile(r"`([^`\n]{3,60})`|(?<![\w/.-])([A-Za-z0-9-]*[A-Za-z0-9][_/.][A-Za-z0-9_/.-]*[A-Za-z0-9])"
                           r"|\b([A-Z][A-Z0-9]*_[A-Z0-9_]+|[A-Z]{4,})\b")
TOOL_SCAN_FILES = 2000   # files read per check at most (bin/ and hooks/ trees)
TOOL_SCAN_BYTES = 262144   # a larger file is not a tool script


def tool_terms(old: str, new: str) -> tuple[set[str], set[str]]:
    """The old item's distinctive terms for restriction_tools: identifiers (_TOOL_TERM_RE) of its
    changed sentences that the new text no longer has, and its rule words (_rule_words) the new text
    dropped (all of the changed sentences' when it dropped fewer than two)."""
    from .coordinator import _changed_sentences, _rule_words
    changed = " ".join(_changed_sentences(old, new)) or old
    ids = {next(g for g in m.groups() if g) for m in _TOOL_TERM_RE.finditer(changed)}
    ids = {t for t in ids if t not in new and not re.fullmatch(r"[\d./-]+", t)}
    words = _rule_words(changed) - _rule_words(new)
    return ids, (words if len(words) >= 2 else _rule_words(changed))


def tool_hit(line: str, ids: set[str], words: set[str]) -> bool:
    """A line names an identifier of the old item, or two of its rule words (as word starts)."""
    if any(t in line for t in ids):
        return True
    low = line.lower()
    return sum(bool(re.search(rf"\b{re.escape(w)}", low)) for w in words) >= 2


def replaced_items(body: str, text: str) -> list[tuple[str, str]]:
    """The items of a Restrictions section a `replaces` retired that its new `text` loosens or drops,
    as (old item, the new item it became or ""). An item equal to a new item, compared loosely, is
    left out; one only contained in a new item (extended with an exception) is not. Any other item
    is paired with the new item that shares the most of its rule words (_widens), and judged as a
    quoted edit to it (_loosens); an item with no such new item was dropped."""
    from .coordinator import _items, _loose, _loosens, _rule_words, _widens
    olds = [" ".join(body[s:e].split()) for s, e, _ in _items(body)]
    news = [" ".join(text[s:e].split()) for s, e, _ in _items(text)]
    kept = {_loose(n) for n in news}
    out = []
    for old in olds:
        if not old or _loose(old) in kept:
            continue
        near = [n for n in news if _widens(old, _rule_words(n))]
        new = max(near, key=lambda n: len(_rule_words(old) & _rule_words(n)), default="")
        if _loosens(old, new):
            out.append((old, new))
    return out


def restriction_tools(p: Project, loosened: list[tuple[str, str, str]], batch: list[int]) -> list[str]:
    """After Restrictions items were loosened or removed, name the project's own tools that still carry
    the old wording: files under bin/ and hooks/ (project root and harness), harness/schedules.json,
    and the command and heal check of each schedule, whose lines name an identifier of the old item
    or two of its dropped rule words (tool_terms). `loosened`: (old item, new text, over), `over`
    empty for the user's word, else the end that retired the item. A file a task of this batch
    names (its title or spec has the file's path or name) is left out. One queued restriction_tools
    event per item with such files, for the next digest: model-free, it edits nothing. Returns the
    event texts."""
    from .coordinator import clip
    files: list[tuple[str, str]] = []   # (label, text)
    seen = 0
    for base in (p.root, p.harness):
        for sub in ("bin", "hooks"):
            top = base / sub
            if not top.is_dir():
                continue
            for f in sorted(top.rglob("*")):
                if seen >= TOOL_SCAN_FILES:
                    break
                if not f.is_file() or any(x in f.parts for x in (".git", "__pycache__", "node_modules")):
                    continue
                seen += 1
                try:
                    if f.stat().st_size > TOOL_SCAN_BYTES:
                        continue
                    raw = f.read_bytes()
                except OSError:
                    continue
                if b"\0" not in raw:
                    files.append((str(f.relative_to(p.root)), raw.decode("utf-8", "replace")))
    sched = p.harness / "schedules.json"
    if sched.is_file() and sched.stat().st_size <= TOOL_SCAN_BYTES:
        files.append((str(sched.relative_to(p.root)), sched.read_text(errors="replace")))
    for r in p.db.q("SELECT name, payload FROM schedules"):
        payload = json.loads(r["payload"] or "{}")
        heal_block = payload.get("heal") or {}
        cmds = [payload.get("command")] + [heal_block.get(k) for k in ("check", "fix", "suppress")]
        files.append((f"schedule {r['name']}", "\n".join(str(c) for c in cmds if c)))
    named = " ".join(f"{t['title']} {t['spec']}" for t in (p.db.task(i) for i in batch) if t)
    out = []
    for old, new, over in loosened:
        ids, words = tool_terms(old, new)
        hits = []
        for label, text in files:
            line = next((n for n, x in enumerate(text.splitlines(), 1) if tool_hit(x, ids, words)), 0)
            if not line or label in named or (not label.startswith("schedule ") and Path(label).name in named):
                continue
            hits.append(label + (f":{line}" if not label.startswith("schedule ") else ""))
        if not hits:
            continue
        x = ((f"`over` retired the Restrictions item \"{clip(old, 200)}\" ({clip(over, 150)})" if over
              else f"The user's word loosened the Restrictions item \"{clip(old, 200)}\"")
             + f", but these project tools still name its terms and no task of that batch names them: "
             + ", ".join(hits[:10])
             + (f" and {len(hits) - 10} more" if len(hits) > 10 else "")
             + ". Queue a priority-1 task updating every tool that enforces the old wording, or say in "
               "`summary` why these do not enforce it.")
        out.append(x)
        p.db.x("INSERT INTO events(ts,source,kind,severity,text,status) VALUES(?,?,?,?,?,?)",
               (time.time(), "harness", "restriction_tools", "normal", x, "queued"))
    return out


# The daily review's 7-day count per Restrictions item: evidence for the coordinator's one-time ask
# to drop or loosen a rule that only blocks work. Model-free; it never edits the charter or asks.
CITE_WINDOW_S = 7 * 86400
BLOCKS_MIN = 3   # review.rule_blocks_min: blocked-work citations that list an item


def citations(p: Project, now: float | None = None, window_s: float = CITE_WINDOW_S) -> list[dict]:
    """Per item of the charter's Restrictions sections, in charter order: {item, blocked, asks,
    stopped} over the last `window_s`. `blocked`: task_blocked events (a run that ended blocked, or a
    task the daemon blocked) citing the item; `asks`: asks sent to the user citing it; `stopped`:
    coordinator actions the guard or relay refused (rejected_actions; the ask gate's ask_user refusals
    are left out) citing it. A text cites an item when it is about one of the item's sentences
    (coordinator._widens on its _rule_words); one text may cite several items."""
    from .coordinator import _items, _rule_words, _sentences, _widens
    from .prompts import charter_restrictions
    try:
        body = charter_restrictions(p.charter_path.read_text())
    except OSError:
        return []
    items = [" ".join(body[s:e].split()) for s, e, _ in _items(body)]
    rows = [{"item": x, "sentences": _sentences([x]), "blocked": 0, "asks": 0, "stopped": 0} for x in items if x]
    if not rows:
        return []
    since = (time.time() if now is None else now) - window_s
    texts: list[tuple[str, str]] = []
    texts += [("blocked", e["text"]) for e in p.db.q(
        "SELECT text FROM events WHERE kind='task_blocked' AND ts>=?", (since,))]
    texts += [("asks", m["text"]) for m in p.db.q(
        "SELECT text FROM messages WHERE direction='out' AND kind='ask' AND ts>=?", (since,))]
    for e in p.db.q("SELECT text, data FROM events WHERE kind='rejected_actions' AND ts>=?", (since,)):
        try:
            got = json.loads(e["data"]) if e["data"] else None
        except ValueError:
            got = None
        texts += [("stopped", str(x)) for x in (got if isinstance(got, list) else [e["text"]])
                  if not str(x).startswith("ask_user:")]
    for key, text in texts:
        words = _rule_words(text or "")
        for r in rows:
            if any(_widens(s, words) for s in r["sentences"]):
                r[key] += 1
    return [{k: v for k, v in r.items() if k != "sentences"} for r in rows]


def only_blocks_lines(p: Project, now: float | None = None, least: int = BLOCKS_MIN) -> list[str]:
    """The self-efficiency audit's lines: one per Restrictions item whose blocked runs and asks
    reached `least` in 7 days while it stopped no banned action; [] when none qualify."""
    from .coordinator import clip
    out = []
    for r in citations(p, now):
        held = r["blocked"] + r["asks"]
        if held >= least and not r["stopped"]:
            out.append(f"Restrictions item only blocked work in {CITE_WINDOW_S // 86400} d (held {r['blocked']} "
                       f"blocked run{'s' * (r['blocked'] != 1)} and {r['asks']} ask{'s' * (r['asks'] != 1)}, "
                       f"stopped no banned action): \"{clip(r['item'], 160)}\"")
    return out
