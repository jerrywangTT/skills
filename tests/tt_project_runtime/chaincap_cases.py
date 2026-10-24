# Continue-chain (chaincap) audit tests of tests/test_tt_project_runtime.py, which runs this file in its
# own namespace (see _include_cases there). Add new continue-chain tests here, not there.
from __future__ import annotations


def _chain_hop(p, prev, title, usd, labels=(), parent=None):
    tid = p.db.add_task(title, "s", kind="code", origin="daemon", parent=parent,
                        labels=list(labels) or [f"continues:{prev}"])
    p.db.update_task(tid, status="done", spent_usd=usd)
    return tid


def _chain_root(p, title, usd):
    tid = p.db.add_task(title, "s", kind="code", origin="user")
    p.db.update_task(tid, status="done", spent_usd=usd)
    return tid


def test_continue_chain_audit_counts_review_fix_hops_apart_from_continuations(env):
    """A review fix (the daemon's review_fix: label, or a 'Fix review #N' title) is no continuation: the
    review-loop line counts its spend, so the chain audit shows the two kinds as separate figures. A
    `continue:` or FOLLOWS hop that took a failed review's findings over (review_fix: added later) stays a
    continuation."""
    import time
    from ttp import audit, chaincap
    p = make(env)
    since = time.time() - 60
    # Pure review-fix lineage: two fix rounds, one by label, one the coordinator added by title only.
    fr = _chain_root(p, "fix lineage", 2.0)
    f1 = _chain_hop(p, fr, "Fix review #901: fix lineage", 0.5, [f"continues:{fr}", "review_fix:901"])
    f2 = _chain_hop(p, f1, "Fix review #902: fix lineage", 0.25)
    # Pure continuation lineage: a continue: hop by parent, a FOLLOWS hop, and a continue: hop that took
    # a review's findings over.
    cr = _chain_root(p, "split lineage", 3.0)
    c1 = _chain_hop(p, cr, "continue: split lineage, part 2", 1.0, ["x"], parent=cr)
    c2 = _chain_hop(p, c1, "split lineage, part 3", 1.5, [f"{chaincap.FOLLOWS}{c1}"])
    c3 = _chain_hop(p, c2, "continue: split lineage, part 4", 0.75, [f"continues:{c2}", "review_fix:903"])
    # Mixed lineage: a continuation, then a fix of its review.
    mr = _chain_root(p, "mixed lineage", 4.0)
    m1 = _chain_hop(p, mr, "continue: mixed lineage", 2.0)
    m2 = _chain_hop(p, m1, "Fix review #904: mixed lineage", 0.5, [f"continues:{m1}", "review_fix:904"])
    by = {x["root"]: x for x in chaincap.lineages(p.db, since)}
    assert {k: by[fr][k] for k in ("hops", "fix_hops", "fix_usd", "cont_hops", "cont_usd", "usd", "tip")} == \
        {"hops": 2, "fix_hops": 2, "fix_usd": 0.75, "cont_hops": 0, "cont_usd": 0.0, "usd": 2.75, "tip": f2}
    assert {k: by[cr][k] for k in ("hops", "fix_hops", "fix_usd", "cont_hops", "cont_usd", "usd", "tip")} == \
        {"hops": 3, "fix_hops": 0, "fix_usd": 0.0, "cont_hops": 3, "cont_usd": 3.25, "usd": 6.25, "tip": c3}
    assert {k: by[mr][k] for k in ("hops", "fix_hops", "fix_usd", "cont_hops", "cont_usd", "usd", "tip")} == \
        {"hops": 2, "fix_hops": 1, "fix_usd": 0.5, "cont_hops": 1, "cont_usd": 2.0, "usd": 6.5, "tip": m2}
    line = chaincap.audit_line(p.db, since)
    assert line.startswith("continue chains: 3 lineage(s), 7 hops, $15.50 with their first tasks; "
                           "continuations: 4 hops in 2 lineage(s), $5.25; "
                           "review fixes (also in the review-loop line): 3 hops in 2 lineage(s), $1.25; "), line
    assert f"#{fr} fix lineage: 2 hops (2 review fixes) to #{f2}" in line
    assert f"#{cr} split lineage: 3 hops to #{c3}" in line
    assert f"#{mr} mixed lineage: 2 hops (1 review fix) to #{m2}" in line
    # The 'tasks continued' line lists continuations; review fixes are one separate figure.
    assert [t["id"] for t in audit.continued_tasks(p.db, since)] == [m1, c3]
    assert sorted(t["id"] for t in audit.continued_tasks(p.db, since, fixes=True)) == sorted([f1, f2, m2])
    out = "\n".join(audit.lines(p.db))
    assert "tasks continued from an earlier one: 2 (+ 3 review fixes, $1.25, counted in the review-loop line): " \
           f"#{m1} continue: mixed lineage continues #{mr}" in out, out
    assert set(audit.report(p.db)) >= {"continued_tasks", "review_fix_tasks"}


def test_review_fix_kind_reads_labels_and_falls_back_to_the_fix_title():
    from ttp import chaincap
    fix = chaincap.is_review_fix
    assert fix({"title": "anything", "labels": '["continues:1", "review_fix:2"]'})
    assert fix({"title": "  fix review #12: x", "labels": '["continues:1"]'})
    assert not fix({"title": "continue: x", "labels": '["continues:1", "review_fix:2"]'})
    assert not fix({"title": "x", "labels": '["follows:1", "review_fix:2"]'})
    assert not fix({"title": "Fix reviewer docs", "labels": '["continues:1"]'})
    assert not fix({"title": "x", "labels": "not json"}) and not fix({"title": "x", "labels": '{"review_fix:1": 1}'})
    assert not fix({"title": None, "labels": None})
