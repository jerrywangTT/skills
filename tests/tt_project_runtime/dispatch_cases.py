# Dispatch pick-order and worker-slot tests of tests/test_tt_project_runtime.py, which runs this file in
# its own namespace (see _include_cases there): helpers, fixtures and imports come from that file, and
# pytest collects these tests under its name. Add new tests of dispatch's pick order, its worker slots
# and Daemon.check_slot_wait here.
from __future__ import annotations


def _slot_setup(env, monkeypatch, tmp_path, slots):
    """A project with `slots` worker slots whose worker runs stay running until release() (the
    stand-in agent waits for a file), and a daemon. Returns (p, d, release, settle)."""
    from ttp.daemon import Daemon
    p = make(env)
    _no_events(p)
    p.db.x("UPDATE tasks SET status='done'")
    p.set_config("budget.max_parallel_workers", slots)
    gate = tmp_path / "release-runs"
    monkeypatch.setenv("TTP_FAKE_SHELL", f'while [ ! -e "{gate}" ]; do sleep 0.05; done')
    d = Daemon(p.base)
    d.cfg = p.config()
    d.update_gates()

    def release():
        gate.touch()

    def settle(daemon):
        """Release every run and reap it with `daemon`, then hold new runs again."""
        release()
        deadline = time.time() + 30
        while _slot_tasks(p) and time.time() < deadline:
            daemon.reap_runs()
            time.sleep(0.05)
        assert not _slot_tasks(p), "a stand-in run did not end"
        gate.unlink()
    return p, d, release, settle


def _slot_tasks(p):
    return [r["task"] for r in p.db.q("SELECT task FROM runs WHERE role!='coordinator' AND status='running' "
                                      "ORDER BY id")]


def _owe(p, tid, due):
    from ttp import obligations
    return obligations.add(p, f"deliverable of #{tid}", owners=[tid],
                           milestones=[{"key": "m1", "due": due, "evidence_required": "a result"}])


def test_dispatch_starts_obligation_owners_then_priority_1_first_and_never_past_max_parallel(env, monkeypatch,
                                                                                            tmp_path):
    p, d, release, settle = _slot_setup(env, monkeypatch, tmp_path, slots=2)
    try:
        now = time.time()
        old_p3 = p.db.add_task("older routine work", "s", kind="work", tier="light", origin="user", priority=3)
        plain_p1 = p.db.add_task("urgent work", "s", kind="work", tier="light", origin="user", priority=1)
        late_owner = p.db.add_task("owes a later deliverable", "s", kind="work", tier="light", origin="user",
                                   priority=3)
        early_owner = p.db.add_task("owes the next deliverable", "s", kind="work", tier="light", origin="user",
                                    priority=3)
        _owe(p, late_owner, now + 7200)
        _owe(p, early_owner, now + 3600)
        assert "## Obligation o" in p.db.task(early_owner)["spec"], "attached through the real record path"
        assert [t["id"] for t in p.db.ready_tasks()] == [early_owner, late_owner, plain_p1, old_p3]
        d.dispatch()
        assert _slot_tasks(p) == [early_owner, late_owner], "obligation owners first, the earliest due first"
        d.dispatch()
        assert len(_slot_tasks(p)) == 2, "a second pass started past max_parallel"
        settle(d)
        d.dispatch()
        assert _slot_tasks(p) == [plain_p1, old_p3], "then priority 1 before the older priority 3"
    finally:
        release()


def test_pick_order_after_a_daemon_restart_comes_from_the_task_records(env, monkeypatch, tmp_path):
    from ttp.daemon import Daemon
    p, d, release, settle = _slot_setup(env, monkeypatch, tmp_path, slots=1)
    try:
        busy = p.db.add_task("holds the only slot", "s", kind="work", tier="light", origin="user", priority=3)
        d.dispatch()
        assert _slot_tasks(p) == [busy]
        first = p.db.add_task("first in line before the restart", "s", kind="work", tier="light", origin="user",
                              priority=1)
        later = p.db.add_task("queued later", "s", kind="work", tier="light", origin="user", priority=2)
        d.dispatch()
        assert _slot_tasks(p) == [busy] and [t["id"] for t in p.db.ready_tasks()] == [first, later]
        # While the daemon is down, the later task becomes an obligation's owner.
        _owe(p, later, time.time() + 3600)
        settle(d)
        restarted = Daemon(p.base)
        restarted.cfg = p.config()
        restarted.update_gates()
        restarted.dispatch()
        assert _slot_tasks(p) == [later], "the restarted daemon kept a stale pick order"
    finally:
        release()


def _slot_wait_events(p):
    return p.db.q("SELECT * FROM events WHERE kind='observation' AND fingerprint LIKE 'slot_wait:%' ORDER BY id")


def test_slot_wait_fires_once_per_urgent_task_only_while_every_slot_is_taken(env, monkeypatch, tmp_path):
    p, d, release, settle = _slot_setup(env, monkeypatch, tmp_path, slots=1)
    try:
        holder = p.db.add_task("routine work holding the slot", "s", kind="work", tier="light", origin="user",
                               priority=3)
        d.dispatch()
        assert _slot_tasks(p) == [holder]
        owner = p.db.add_task("owes a deliverable", "s", kind="work", tier="light", origin="user", priority=2)
        oid = _owe(p, owner, time.time() + 3600)
        urgent = p.db.add_task("urgent work", "s", kind="work", tier="light", origin="user", priority=1)
        routine = p.db.add_task("more routine work", "s", kind="work", tier="light", origin="user", priority=3)
        d.dispatch()
        assert _slot_tasks(p) == [holder], "a full slot let a task start"
        start = time.time()

        def check(minutes):
            d._slot_wait_checked = start + minutes * 60 - 60
            monkeypatch.setattr(time, "time", lambda: start + minutes * 60)
            d.check_slot_wait()
            return _slot_wait_events(p)

        assert check(0) == [] and check(5) == [] and check(10) == [] and check(14) == []
        events = check(15)
        assert sorted(e["task"] for e in events) == [owner, urgent], "only obligation owners and priority 1"
        by_task = {e["task"]: e for e in events}
        assert by_task[owner]["severity"] == "high" and by_task[urgent]["severity"] == "normal"
        text = by_task[owner]["text"]
        assert f"#{owner} " in text and oid in text and "all 1 are taken" in text
        assert f"#{holder} routine work holding the slot (p3, run " in text, "the slot holders with priorities"
        assert routine not in [e["task"] for e in events]
        assert len(check(20)) == 2 and len(check(60)) == 2, "raised again for the same wait"
        monkeypatch.undo()
        # A free slot: dispatch would start them, so nothing is raised however long they stay queued.
        settle(d)
        for t in (owner, urgent):
            p.db.update_task(t, status="queued")
        p.db.x("DELETE FROM events WHERE fingerprint LIKE 'slot_wait:%'")
        start = time.time()
        assert all(check(m) == [] for m in (0, 5, 10, 15, 20, 30)), "raised while a slot was free"
    finally:
        release()


def test_slot_wait_is_off_at_zero(env, monkeypatch, tmp_path):
    p, d, release, settle = _slot_setup(env, monkeypatch, tmp_path, slots=1)
    try:
        p.db.add_task("routine", "s", kind="work", tier="light", origin="user", priority=3)
        d.dispatch()
        p.db.add_task("urgent work", "s", kind="work", tier="light", origin="user", priority=1)
        d.cfg["coordinator"]["urgent_slot_wait_min"] = 0
        start = time.time()
        for minutes in (0, 5, 10, 15, 20):
            d._slot_wait_checked = start + minutes * 60 - 60
            monkeypatch.setattr(time, "time", lambda: start + minutes * 60)
            d.check_slot_wait()
        assert _slot_wait_events(p) == []
    finally:
        release()


def test_a_task_waiting_on_an_ended_detached_driver_holds_no_worker_slot(env, monkeypatch, tmp_path):
    """Only live model runs count against max_parallel: a task whose run handed off waiting on its
    detached driver, which then ended, is queued and frees its slot at once (no detach-record change
    needed)."""
    p, d, release, settle = _slot_setup(env, monkeypatch, tmp_path, slots=1)
    try:
        rc = tmp_path / "drv.rc"
        rc.write_text("0\n")   # the driver ended
        monkeypatch.delenv("TTP_FAKE_SHELL")
        monkeypatch.setenv("TTP_FAKE_RESULT", json.dumps({"status": "waiting", "summary": "driver runs",
                                                          "retry_when": f"ttp detach --check {rc}",
                                                          "retry_after_s": 3600}))
        waiter = p.db.add_task("waits on its driver", "s", kind="work", tier="light", origin="user", priority=3)
        d.dispatch()
        assert _slot_tasks(p) == [waiter]
        deadline = time.time() + 30
        while _slot_tasks(p) and time.time() < deadline:
            d.reap_runs()
            time.sleep(0.05)
        t = p.db.task(waiter)
        assert t["status"] == "queued" and json.loads(t["result"])["status"] == "waiting"
        monkeypatch.delenv("TTP_FAKE_RESULT")
        monkeypatch.setenv("TTP_FAKE_SHELL", f'while [ ! -e "{tmp_path / "release-runs"}" ]; do sleep 0.05; done')
        nxt = p.db.add_task("next in line", "s", kind="work", tier="light", origin="user", priority=3)
        d.dispatch()
        assert _slot_tasks(p) == [nxt], "the waiting task still held the only slot"
    finally:
        release()
