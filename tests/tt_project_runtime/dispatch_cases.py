# Dispatch pick-order and worker-slot tests of tests/test_tt_project_runtime.py, which runs this file in
# its own namespace (see _include_cases there): helpers, fixtures and imports come from that file, and
# pytest collects these tests under its name. Add new tests of dispatch's pick order, its worker slots
# and Daemon.check_slot_wait here.
from __future__ import annotations


def _slot_setup(env, monkeypatch, tmp_path, slots):
    """A project with `slots` worker slots whose worker runs stay running until drain() (the
    stand-in agent waits for a file), and a daemon. Returns (p, d, drain, settle)."""
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

    def drain(daemon):
        """Release every run and reap it with `daemon` until no worker run is running. Safe in a
        failed test's cleanup: it asserts nothing and its deadline ignores a patched time.time."""
        gate.touch()
        deadline = time.monotonic() + 30
        while _slot_tasks(p) and time.monotonic() < deadline:
            daemon.reap_runs()
            time.sleep(0.05)

    def settle(daemon):
        """Release every run and reap it with `daemon`, then hold new runs again."""
        drain(daemon)
        assert not _slot_tasks(p), "a stand-in run did not end"
        gate.unlink()
    return p, d, drain, settle


def _slot_tasks(p):
    return [r["task"] for r in p.db.q("SELECT task FROM runs WHERE role!='coordinator' AND status='running' "
                                      "ORDER BY id")]


def _owe(p, tid, due):
    from ttp import obligations
    return obligations.add(p, f"deliverable of #{tid}", owners=[tid],
                           milestones=[{"key": "m1", "due": due, "evidence_required": "a result"}])


def test_dispatch_starts_obligation_owners_then_priority_1_first_and_never_past_max_parallel(
        env, monkeypatch, tmp_path, reap_tmp_processes):
    p, d, drain, settle = _slot_setup(env, monkeypatch, tmp_path, slots=2)
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
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"


def test_pick_order_after_a_daemon_restart_comes_from_the_task_records(env, monkeypatch, tmp_path,
                                                                       reap_tmp_processes):
    from ttp.daemon import Daemon
    p, d, drain, settle = _slot_setup(env, monkeypatch, tmp_path, slots=1)
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
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"


def _slot_wait_events(p):
    return p.db.q("SELECT * FROM events WHERE kind='observation' AND fingerprint LIKE 'slot_wait:%' ORDER BY id")


def test_slot_wait_fires_once_per_urgent_task_only_while_every_slot_is_taken(env, monkeypatch, tmp_path,
                                                                             reap_tmp_processes):
    p, d, drain, settle = _slot_setup(env, monkeypatch, tmp_path, slots=1)
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
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"


def test_slot_wait_is_off_at_zero(env, monkeypatch, tmp_path, reap_tmp_processes):
    p, d, drain, settle = _slot_setup(env, monkeypatch, tmp_path, slots=1)
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
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"


def test_a_task_waiting_on_an_ended_detached_driver_holds_no_worker_slot(env, monkeypatch, tmp_path,
                                                                         reap_tmp_processes):
    """Only live model runs count against max_parallel: a task whose run handed off waiting on its
    detached driver, which then ended, is queued and frees its slot at once (no detach-record change
    needed)."""
    p, d, drain, settle = _slot_setup(env, monkeypatch, tmp_path, slots=1)
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
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"


def _board_setup(env, monkeypatch, tmp_path, board):
    """_slot_setup with three worker slots and `board` slots of resource board. Each stand-in run hands
    off $G/result-<task>.json when the test wrote one, else runs until $G/go-<task> or drain(). Returns
    (p, d, drain, go): go(tid) ends that task's run and reaps it."""
    p, d, drain, settle = _slot_setup(env, monkeypatch, tmp_path, slots=3)
    g = shlex.quote(str(tmp_path))
    monkeypatch.setenv("TTP_FAKE_SHELL", f'G={g}; if [ -e "$G/result-$TTP_TASK.json" ]; then '
                                         f'cp "$G/result-$TTP_TASK.json" "$TTP_RUN_DIR/result.json"; exit 0; fi; '
                                         f'while [ ! -e "$G/go-$TTP_TASK" ] && [ ! -e "$G/release-runs" ]; '
                                         f'do sleep 0.05; done')
    _board_slots(p, d, board)

    def go(tid):
        (tmp_path / f"go-{tid}").touch()
        assert _until_true(lambda: d.reap_runs() or tid not in _slot_tasks(p)), f"the run of #{tid} did not end"
    return p, d, drain, go


def _board_slots(p, d, n, **more):
    p.set_config("resources", {"board": n, **more})
    d.cfg = p.config()


def _board_add(p, title, priority, resources=("board",)):
    from ttp import coordinator as coord
    assert coord.apply(p, [{"type": "task_add", "title": title, "spec": "s", "kind": "work", "tier": "light",
                            "resources": list(resources), "exclusive": True, "priority": priority}]) == []
    return p.db.one("SELECT id FROM tasks WHERE title=?", (title,))["id"]


def _board_waiting(p, d, tmp_path, tid, probe="ttp lock --probe board"):
    """Start task tid (the test leaves a slot of each of its resources free), let its run hand off
    waiting on `probe` and reap it; the next run of the task stays running until go(tid)."""
    res = tmp_path / f"result-{tid}.json"
    res.write_text(json.dumps({"status": "waiting", "summary": "board busy", "retry_when": probe,
                               "retry_after_s": 3600}))
    d.dispatch()
    assert _until_true(lambda: d.reap_runs() or p.db.task(tid)["status"] == "queued" and tid not in _slot_tasks(p))
    res.unlink()
    t = p.db.task(tid)
    assert json.loads(t["result"])["status"] == "waiting" and t["not_before"] > time.time() + 3000, t


def test_a_freed_exclusive_slot_goes_to_the_higher_priority_task_waiting_for_it(env, monkeypatch, tmp_path,
                                                                                reap_tmp_processes):
    """The reported case: a priority-3 run holds board, a priority-1 task waits on `ttp lock --probe board`
    (retry timer an hour off) and a newer priority-3 task is ready. Once board frees, dispatch starts the
    waiting priority-1 task, not the ready one."""
    p, d, drain, go = _board_setup(env, monkeypatch, tmp_path, board=2)
    try:
        a = _board_add(p, "A holds board", 3)
        p1 = _board_add(p, "P1 urgent", 1)
        _board_waiting(p, d, tmp_path, p1)
        assert _slot_tasks(p) == [a]
        _board_slots(p, d, 1)
        c = _board_add(p, "C routine", 3)
        d.dispatch()
        assert _slot_tasks(p) == [a], "a task started on a full board"
        go(a)
        d.dispatch()
        assert _slot_tasks(p) == [p1], "the freed slot went to the lower-priority ready task"
        assert p.db.task(c)["status"] == "queued"
    finally:
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"


def test_a_freed_exclusive_slot_goes_to_the_higher_priority_of_two_ready_tasks(env, monkeypatch, tmp_path,
                                                                               reap_tmp_processes):
    p, d, drain, go = _board_setup(env, monkeypatch, tmp_path, board=1)
    try:
        a = _board_add(p, "A holds board", 3)
        d.dispatch()
        assert _slot_tasks(p) == [a]
        c = _board_add(p, "C routine, older", 3)
        p1 = _board_add(p, "P1 urgent", 1)
        d.dispatch()
        assert _slot_tasks(p) == [a]
        go(a)
        d.dispatch()
        assert _slot_tasks(p) == [p1] and p.db.task(c)["status"] == "queued"
    finally:
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"


def test_a_waiting_higher_priority_task_reserves_the_slot_against_ttp_lock_and_lower_tasks(
        env, monkeypatch, tmp_path, reap_tmp_processes):
    """board has two slots: run A holds one, a test `ttp lock` the other, P1 waits. Once the lock ends,
    the free slot is reserved for P1: another `ttp lock` waits (75, naming P1) and dispatch starts P1,
    not the lower-priority ready C."""
    from ttp import locks
    p, d, drain, go = _board_setup(env, monkeypatch, tmp_path, board=2)
    flag, marks = tmp_path / "unlock", tmp_path / "marks"
    holder = None
    try:
        a = _board_add(p, "A holds board", 3)
        p1 = _board_add(p, "P1 urgent", 1)
        _board_waiting(p, d, tmp_path, p1)
        holder = _lock(_lock_env(env, p, TTP_TASK="900"), "board", cmd=_mark(marks, "H", flag))
        assert _until_true(lambda: marks.exists() and "start H" in marks.read_text())
        c = _board_add(p, "C routine", 3)
        d.dispatch()
        assert _slot_tasks(p) == [a]
        assert locks.reserved_by(d._reserve_path("board")) == f"task #{p1}"
        flag.touch()
        assert _ended([holder]) == [0]
        r = subprocess.run([sys.executable, str(TTP), "lock", "--timeout", "2", "board", "--", "true"],
                           env=_lock_env(env, p, TTP_TASK="901"), capture_output=True, text=True, timeout=60)
        assert r.returncode == 75 and f"reserved for task #{p1}" in r.stderr, (r.returncode, r.stderr)
        d.dispatch()
        assert _slot_tasks(p) == [a, p1] and p.db.task(c)["status"] == "queued"
    finally:
        flag.touch()
        if holder:
            _ended([holder])
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"


def test_a_reservation_that_keeps_a_free_slot_unused_lapses_after_exclusive_reserve_s(
        env, monkeypatch, tmp_path, reap_tmp_processes):
    """P1 needs board and cable; a test `ttp lock` holds cable. P1 reserves the free board, so the lower C
    does not start on it, until the reservation lapses after budget.exclusive_reserve_s (1 s here)."""
    from ttp import locks
    p, d, drain, go = _board_setup(env, monkeypatch, tmp_path, board=1)
    flag, marks = tmp_path / "unlock", tmp_path / "marks"
    holder = None
    try:
        p.set_config("budget.exclusive_reserve_s", 1)
        _board_slots(p, d, 1, cable=1)
        p1 = _board_add(p, "P1 urgent", 1, resources=("board", "cable"))
        _board_waiting(p, d, tmp_path, p1, probe="ttp lock --probe board,cable")
        holder = _lock(_lock_env(env, p, TTP_TASK="900"), "cable", cmd=_mark(marks, "H", flag))
        assert _until_true(lambda: marks.exists() and "start H" in marks.read_text())
        c = _board_add(p, "C routine", 3)
        d.dispatch()
        d.dispatch()
        assert _slot_tasks(p) == [], "C started on a board reserved for P1"
        assert locks.reserved_by(d._reserve_path("board")) == f"task #{p1}"
        time.sleep(1.2)
        d.dispatch()
        assert _slot_tasks(p) == [c], "the reservation never lapsed"
        assert locks.reserved_by(d._reserve_path("board")) != f"task #{p1}"
        assert p.db.task(p1)["status"] == "queued"
    finally:
        flag.touch()
        if holder:
            _ended([holder])
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"


def test_equal_priority_reserves_nothing_and_the_older_waiting_task_takes_the_freed_slot(
        env, monkeypatch, tmp_path, reap_tmp_processes):
    from ttp import locks
    p, d, drain, go = _board_setup(env, monkeypatch, tmp_path, board=2)
    try:
        a = _board_add(p, "A holds board", 3)
        w = _board_add(p, "W waits, older", 3)
        _board_waiting(p, d, tmp_path, w)
        _board_slots(p, d, 1)
        c = _board_add(p, "C ready, newer", 3)
        d.dispatch()
        assert _slot_tasks(p) == [a]
        assert locks.reserved_by(d._reserve_path("board")) is None, "an equal rank took a reservation"
        go(a)
        d.dispatch()
        assert _slot_tasks(p) == [w] and p.db.task(c)["status"] == "queued"
        assert locks.reserved_by(d._reserve_path("board")) is None
    finally:
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"


def test_a_paused_exclusive_resource_is_reserved_for_no_one_and_nothing_starts_on_it(
        env, monkeypatch, tmp_path, reap_tmp_processes):
    from ttp import coordinator as coord, locks
    p, d, drain, go = _board_setup(env, monkeypatch, tmp_path, board=1)
    try:
        p1 = _board_add(p, "P1 urgent", 1)
        _board_waiting(p, d, tmp_path, p1)
        before = p.db.task(p1)["not_before"]
        coord.pause_resource(p, "board", True, reason="maintenance", by="user")
        _board_add(p, "C routine", 3)
        d.dispatch()
        assert _slot_tasks(p) == []
        assert locks.reserved_by(d._reserve_path("board")) is None, "a paused resource was reserved"
        assert p.db.task(p1)["not_before"] == before, "a task on a paused resource was woken"
    finally:
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"


def test_a_reservation_keeps_one_slot_so_lower_tasks_take_the_spare_slots_and_its_start_serves_it(
        env, monkeypatch, tmp_path, reap_tmp_processes):
    """board has two slots and P1 (board and cable) reserves it while cable is held: the lower C takes the
    spare slot, the lower D not the one kept for P1. Once P1 starts, its reservation is served."""
    from ttp import locks
    p, d, drain, go = _board_setup(env, monkeypatch, tmp_path, board=2)
    flag, marks = tmp_path / "unlock", tmp_path / "marks"
    holder = None
    try:
        _board_slots(p, d, 2, cable=1)
        p1 = _board_add(p, "P1 urgent", 1, resources=("board", "cable"))
        _board_waiting(p, d, tmp_path, p1, probe="ttp lock --probe board,cable")
        holder = _lock(_lock_env(env, p, TTP_TASK="900"), "cable", cmd=_mark(marks, "H", flag))
        assert _until_true(lambda: marks.exists() and "start H" in marks.read_text())
        c = _board_add(p, "C routine", 3)
        dd = _board_add(p, "D routine", 3)
        d.dispatch()
        assert _slot_tasks(p) == [c], "the spare slot stayed idle, or the kept one was taken"
        assert locks.reserved_by(d._reserve_path("board")) == f"task #{p1}"
        go(c)
        flag.touch()
        assert _ended([holder]) == [0]
        d.dispatch()
        assert _slot_tasks(p) == [p1, dd], "P1's start did not serve its reservation"
    finally:
        flag.touch()
        if holder:
            _ended([holder])
        drain(d)
        assert reap_tmp_processes(tmp_path) == [], "a stand-in run outlived the test"
