# Detached job tests of tests/test_tt_project_runtime.py, which runs this file in its own namespace
# (see _include_cases there): helpers, fixtures and imports come from that file, and pytest collects
# these tests under its name. Add new `ttp detach` liveness tests here, not there.
from __future__ import annotations


def _ssh_on_path(tmp_path, monkeypatch):
    """An `ssh` on PATH that runs the remote command here with sh, or fails as an unreachable host
    (exit 255) while the returned flag file exists."""
    bindir, down = tmp_path / "sshbin", tmp_path / "ssh-down"
    bindir.mkdir()
    shim = bindir / "ssh"
    shim.write_text(f'#!/bin/sh\n[ -e {shlex.quote(str(down))} ] && exit 255\nfor last; do :; done\nexec sh -c "$last"\n')
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    return down


def _remote_job(tmp_path, **remote):
    """The .rc path of a local job whose detached.json entry carries this remote contract."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    rc = run_dir / "job.rc"
    (run_dir / "detached.json").write_text(json.dumps([{"name": "job", "rc": str(rc), "remote": remote}]))
    return rc


def _dead_pid():
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_a_remote_hosts_changed_boot_id_is_a_reset_verdict(env, tmp_path, monkeypatch):
    from ttp import detach
    from ttp.runner import boot_id
    _ssh_on_path(tmp_path, monkeypatch)
    assert detach.remote_start("box") == boot_id()
    rc = _remote_job(tmp_path, host="box", boot="a-boot-before-the-reset", pid=str(os.getpid()), marker="")
    why = detach.remote_verdict(rc, {})
    assert why and why.startswith("remote reset:") and "box" in why
    rc = _remote_job(tmp_path, host="box", boot=boot_id(), pid=str(os.getpid()), marker="")
    assert detach.remote_verdict(rc, {}) is None, "a live remote pid on the same boot was called dead"


def test_a_remote_pid_gone_without_its_marker_is_a_verdict_only_on_the_second_probe(env, tmp_path, monkeypatch):
    from ttp import detach
    from ttp.runner import boot_id
    _ssh_on_path(tmp_path, monkeypatch)
    pidfile = tmp_path / "driver.pid"
    rc = _remote_job(tmp_path, host="box", boot=boot_id(), pid=str(pidfile), marker=str(tmp_path / "DONE"))
    memo: dict = {}
    assert detach.remote_verdict(rc, memo) is None, "a pid file not written yet is no dead driver"
    pidfile.write_text(f"{_dead_pid()}\n")
    assert detach.remote_verdict(rc, memo) is None, "one probe that reads the pid dead ended the job"
    why = detach.remote_verdict(rc, memo)
    assert why and why.startswith("remote driver gone:") and "DONE is missing" in why
    pidfile.write_text(f"{os.getpid()}\n")   # alive again (a restarted driver): the count starts over
    assert detach.remote_verdict(rc, memo) is None and "dead" not in memo


def test_a_remote_completion_marker_is_never_a_dead_verdict(env, tmp_path, monkeypatch):
    from ttp import detach
    from ttp.runner import boot_id
    _ssh_on_path(tmp_path, monkeypatch)
    (tmp_path / "DONE").write_text("0\n")
    rc = _remote_job(tmp_path, host="box", boot=boot_id(), pid=str(_dead_pid()), marker=str(tmp_path / "DONE"))
    memo: dict = {}
    assert [detach.remote_verdict(rc, memo) for _ in range(4)] == [None] * 4


def test_an_unreachable_remote_host_is_no_proof_of_death(env, tmp_path, monkeypatch):
    from ttp import detach
    down = _ssh_on_path(tmp_path, monkeypatch)
    down.touch()
    assert detach.remote_start("box") == ""
    rc = _remote_job(tmp_path, host="box", boot="a-boot-before-the-reset", pid=str(_dead_pid()), marker="")
    memo: dict = {}
    assert [detach.remote_verdict(rc, memo) for _ in range(4)] == [None] * 4
    # Unreachable at start: the first answer names the boot, and a later change of it is a reset.
    down.unlink()
    rc = _remote_job(tmp_path, host="box", boot="", pid="", marker="")
    memo = {}
    assert detach.remote_verdict(rc, memo) is None and memo["boot"]
    memo["boot"] = "a-boot-before-the-reset"
    assert (detach.remote_verdict(rc, memo) or "").startswith("remote reset:")


def test_remote_pid_and_marker_need_remote_host(env, tmp_path):
    r = subprocess.run([sys.executable, str(TTP), "detach", "job", "--remote-pid", "5", "--", "true"],
                       capture_output=True, text=True, env=dict(os.environ, TTP_RUN_DIR=str(tmp_path)))
    assert r.returncode != 0 and "go with --remote-host" in r.stderr


def test_stop_records_its_reason_before_a_reader_can_see_the_job_gone(tmp_path, monkeypatch):
    from ttp import detach
    rc = _remote_job(tmp_path)
    seen = []
    # A reader that looks the moment the kill lands (its lock is free) must find the stop's reason.
    monkeypatch.setattr(detach.os, "killpg", lambda pid, sig: seen.append(detach.state(rc)))
    detach.stop(rc, 12345, "remote driver gone: test")
    assert seen == [("lost", "remote driver gone: test")]
    assert detach.state(rc) == ("lost", "remote driver gone: test")


def test_stop_records_its_reason_before_the_job_ends(env, tmp_path, monkeypatch):
    """The keeper's loop reads the job's state while its probe thread stops it: a look between the kill
    and the record took the job for one killed with no reason and wrote that, over the remote verdict."""
    from ttp import detach, locks
    from ttp.runner import boot_id
    rc = _remote_job(tmp_path, host="box", boot=boot_id(), pid="", marker="")
    detach.started(rc).write_text("4242\n")
    killed, seen = [], []
    monkeypatch.setattr(locks, "any_free", lambda paths: bool(killed))
    monkeypatch.setattr(detach.os, "killpg", lambda pid, sig: killed.append(pid) or seen.append(detach.state(rc)))
    assert detach.state(rc) == ("running", "")
    why = "remote driver gone: pid 7 on box is dead"
    detach.stop(rc, 4242, why)
    assert killed == [4242] and seen == [("lost", why)], seen
    assert detach.state(rc) == ("lost", why)


def test_a_keeper_ends_a_job_whose_remote_driver_died_and_lets_its_hold_go(env, tmp_path, monkeypatch):
    p = make(env)
    from ttp import detach, hold, locks
    from ttp.daemon import Daemon
    from ttp.runner import boot_id
    down = _ssh_on_path(tmp_path, monkeypatch)
    mine = _board_task(p, "flash")
    run_dir, release, pidfile = p.runs / "51", tmp_path / "release", tmp_path / "driver.pid"
    driver = subprocess.Popen(["sleep", "600"])   # stands in for the driver on the remote host
    pidfile.write_text(f"{driver.pid}\n")
    paths = Daemon(p.base)._slot_paths("board")
    try:
        _exclusive_run_detaching(p, mine["id"], 51, run_dir, _until(release),
                                 opts=("--remote-host", "box", "--remote-pid", str(pidfile),
                                       "--remote-marker", str(tmp_path / "DONE")))
        entry = json.loads((run_dir / "detached.json").read_text())[0]
        assert entry["remote"] == {"host": "box", "boot": boot_id(), "pid": str(pidfile),
                                   "marker": str(tmp_path / "DONE")}
        rc = pathlib.Path(entry["rc"])
        assert hold.read(run_dir)["state"] == "holding" and not locks.any_free(paths)
        # The host stops answering and its driver dies: unreachable proves nothing, the hold stays.
        down.touch()
        driver.kill()
        driver.wait()
        time.sleep(1)
        assert not locks.any_free(paths) and not locks.job_ended(rc), "an unreachable host ended the job"
        down.unlink()
        assert _until_true(lambda: locks.any_free(paths)), "a dead remote driver's hold was kept"
        st, why = detach.state(rc)
        assert st == "lost" and why.startswith("remote driver gone:")
        rec = hold.read(run_dir)
        assert rec["state"] == "released" and "every detached job of the run ended" in rec["why"]
        assert "remote driver gone" in rec["why"]
        assert not rc.exists(), "the local driver was not ended"
    finally:
        release.touch()
        if driver.poll() is None:
            driver.kill()


def test_a_stopped_job_reads_its_verdict_the_moment_it_dies(env, tmp_path, monkeypatch):
    # The keeper reads its jobs' state while its probe thread ends one: a job seen dead before its
    # verdict was written got a generic "killed" reason, and the keeper then exited with the thread.
    from ttp import detach, locks
    rc = _remote_job(tmp_path, host="box", boot="", pid="", marker="")
    job = locks.try_take([locks.job_lock(rc)], "job")
    seen = []

    def killpg(pid, sig):
        job.close()
        seen.append(detach.state(rc))
    monkeypatch.setattr(detach.os, "killpg", killpg)
    detach.stop(rc, 12345, "remote driver gone: pid 7 on box is dead")
    assert seen == [("lost", "remote driver gone: pid 7 on box is dead")]
    assert detach.state(rc) == seen[0]


def _remote_mirror_job(env, tmp_path, script, name="drv"):
    """A `ttp detach --remote` job started from run 61 through the ssh stub: (project, run dir, remote
    folder, probe argv, ssh env)."""
    p = _device_project(env, device=False)
    run_dir = p.runs / "61"
    run_dir.mkdir(parents=True)
    e = _ssh_stub(tmp_path)
    rdir = tmp_path / "remote"
    r = _ttp_run(p, "detach", "--remote", "box-m", "--dir", str(rdir), name, "--", "sh", "-c", script,
                 env={**e, "TTP_RUN_DIR": str(run_dir)})
    assert r.returncode == 0, r.stderr
    return p, run_dir, rdir, _remote_probe(r.stdout), e


def test_stop_notes_its_verdict_before_the_kill_so_a_concurrent_look_never_reports_a_plain_crash(tmp_path, monkeypatch):
    from ttp import detach
    rc = _remote_job(tmp_path)
    detach.started(rc).write_text("4242\n")
    seen = []
    # Another reader (the daemon, a check) looks at the job the instant its process group dies.
    monkeypatch.setattr(detach.os, "killpg", lambda pid, sig: seen.append(detach.state(rc)))
    detach.stop(rc, 4242, "remote driver gone: pid 7 on box is dead")
    assert seen == [("lost", "remote driver gone: pid 7 on box is dead")]
    assert detach.state(rc) == ("lost", "remote driver gone: pid 7 on box is dead")


def test_a_late_crash_guess_never_overwrites_a_recorded_verdict(tmp_path):
    from ttp import detach
    rc = _remote_job(tmp_path)
    detach._mark_lost(rc, "remote reset: the host restarted")
    assert detach._mark_lost_first(rc, "it was killed after it started") == "remote reset: the host restarted"
    assert detach.lost(rc).read_text().strip() == "remote reset: the host restarted"
    (tmp_path / "b").mkdir()
    other = _remote_job(tmp_path / "b")
    assert detach._mark_lost_first(other, "it was killed after it started") == "it was killed after it started"
    assert detach.lost(other).read_text().strip() == "it was killed after it started"


def test_detach_remote_check_mirrors_the_rc_and_reads_it_once_the_host_is_unreachable(env, tmp_path):
    """The reported failure: a remote job's .log and .rc stayed on a host that then reset, so its probe
    returned 255 forever though the job had finished. The probe now mirrors both into the run folder,
    and with ssh failing it reads the mirrored end: 4 for a failure, 0 for a pass, never 255."""
    p, run_dir, rdir, probe, e = _remote_mirror_job(env, tmp_path, "printf 'out\\377\\n'; exit 3")
    assert probe[probe.index("--mirror") + 1] == str(run_dir.resolve()), probe
    meta = json.loads((run_dir / "drv.remote.json").read_text())
    assert meta["host"] == "box-m" and meta["path"] == str(rdir / "drv") and "end" not in meta
    assert _wait_for_path(rdir / "drv.rc")
    first = _ttp_run(p, *probe[1:], env=e)   # the daemon's probe env carries no TTP_RUN_DIR
    assert first.returncode == 4 and "drv: job finished rc=3" in first.stdout, first.stdout
    assert (run_dir / "drv.remote.rc").read_text().strip() == "3"
    assert (run_dir / "drv.remote.log").read_bytes() == b"out\xff\n", "the log tail is mirrored byte for byte"
    assert json.loads((run_dir / "drv.remote.json").read_text())["end"] == {"rc": "3"}
    down = _ttp_run(p, *probe[1:], env={**e, "SSH_STUB_FAIL": "1"})
    assert down.returncode == 4, down.stdout
    assert "job finished rc=3, as mirrored before box-m stopped answering" in down.stdout, down.stdout
    # The check alone with the env's TTP_RUN_DIR mirrors there too; a mirror is never another job's.
    (run_dir / "drv.remote.json").write_text(json.dumps({"host": "other", "path": str(rdir / "drv"), "end": {"rc": "0"}}))
    assert _ttp_run(p, *probe[1:], env={**e, "SSH_STUB_FAIL": "1"}).returncode == 255
    assert _ttp_run(p, "detach", "--check", str(rdir / "drv.rc"), "--mirror", str(run_dir)).returncode == 2
    r = _ttp_run(p, "detach", "--remote", "box-m", "--dir", str(rdir), "ok", "--", "echo", "fine",
                 env={**e, "TTP_RUN_DIR": str(run_dir)})
    assert r.returncode == 0 and _wait_for_path(rdir / "ok.rc"), r.stderr
    ok = _remote_probe(r.stdout)
    assert _ttp_run(p, *ok[1:], env=e).returncode == 0
    assert _ttp_run(p, *ok[1:], env={**e, "SSH_STUB_FAIL": "1"}).returncode == 0


def test_detach_remote_mirror_keeps_a_bounded_log_tail_sent_only_when_it_grew(env, tmp_path, monkeypatch):
    from ttp import cli, detach
    big = detach.MIRROR_TAIL_BYTES * 3
    p, run_dir, rdir, probe, e = _remote_mirror_job(
        env, tmp_path, f"head -c {big} /dev/zero | tr '\\0' a; echo END; exit 0")
    assert _wait_for_path(rdir / "drv.rc")
    for k, v in e.items():
        monkeypatch.setenv(k, v)
    sent = []
    real = cli._ssh_sh
    monkeypatch.setattr(cli, "_ssh_sh", lambda *a, **k: sent.append(real(*a, **k)) or sent[-1])
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli._detach_check_remote("box-m", [str(rdir / "drv")], mirror=str(run_dir)) == 0
    tail = (run_dir / "drv.remote.log").read_bytes()
    assert len(tail) == detach.MIRROR_TAIL_BYTES and tail == (rdir / "drv.log").read_bytes()[-len(tail):]
    assert tail.endswith(b"END\n")
    assert len(sent[-1].stdout) < detach.MIRROR_TAIL_BYTES * 2, "more than the tail crossed ssh"
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli._detach_check_remote("box-m", [str(rdir / "drv")], mirror=str(run_dir)) == 0
    assert "tail same" in sent[-1].stdout and len(sent[-1].stdout) < 200, "an unchanged log was sent again"
    assert (run_dir / "drv.remote.log").read_bytes() == tail


def test_detach_remote_mirror_unreachable_before_the_end_stays_255_and_the_keeper_refreshes(env, tmp_path, monkeypatch):
    """A host unreachable before any end was mirrored proves nothing: 255 (asleep), naming the log tail
    kept so far. The hold keeper's probe round refreshes the run's mirrors, so the end is caught while
    the host still answers even with no check running."""
    from ttp import detach, hold
    go = tmp_path / "go"
    p, run_dir, rdir, probe, e = _remote_mirror_job(
        env, tmp_path, f"echo step1; while [ ! -e {go} ]; do sleep 0.1; done; echo step2; exit 5")
    pid = int((rdir / "drv.pid").read_text())
    try:
        assert _wait_for_path(rdir / "drv.log") and _until_true(lambda: (rdir / "drv.log").read_text().strip())
        assert _ttp_run(p, *probe[1:], env=e).returncode == 1
        assert (run_dir / "drv.remote.log").read_text() == "step1\n"
        down = _ttp_run(p, *probe[1:], env={**e, "SSH_STUB_FAIL": "1"})
        assert down.returncode == 255 and "ssh to box-m failed" in down.stdout, down.stdout
        assert "last log tail mirrored here" in down.stdout and not (run_dir / "drv.remote.rc").exists()
        go.touch()
        assert _wait_for_path(rdir / "drv.rc")
    finally:
        _kill_driver(pid)
    for k, v in e.items():
        monkeypatch.setenv(k, v)
    hold._remote_round([], {}, [], run_dir)
    assert (run_dir / "drv.remote.rc").read_text().strip() == "5"
    assert (run_dir / "drv.remote.log").read_text() == "step1\nstep2\n"
    monkeypatch.setenv("SSH_STUB_FAIL", "1")
    detach.mirror_refresh(run_dir)   # an ended mirror is not asked again; an unreachable host changes nothing
    down = _ttp_run(p, *probe[1:], env={**e, "SSH_STUB_FAIL": "1"})
    assert down.returncode == 4 and "job finished rc=5, as mirrored" in down.stdout, down.stdout


def test_detach_mirror_probe_reads_the_verdict_before_the_tail(env, tmp_path, monkeypatch):
    """A job that writes its last lines after the verdict was read must still have them in the mirrored
    tail, and the verdict stays the answer's last line."""
    from ttp import cli, detach
    e = _ssh_stub(tmp_path)
    for k, v in e.items():
        monkeypatch.setenv(k, v)
    job = tmp_path / "job"
    (tmp_path / "job.log").write_text("early\n")
    # The fake check reports the job running, then the job writes its last line (as if between steps).
    monkeypatch.setattr(cli, "_REMOTE_CHECK", f'echo "running 1"\necho late >>{tmp_path}/job.log\n')
    d = tmp_path / "mirror"
    d.mkdir()
    r, lines = detach.mirror_probe("box-m", str(job), d)
    assert r.returncode == 0 and lines[-1] == "running 1", lines
    assert (d / "job.remote.log").read_text() == "early\nlate\n", "the tail was taken before the verdict"


def test_a_relative_rc_never_leaves_a_lost_marker_in_the_callers_checkout(env, tmp_path, monkeypatch):
    """A probe run from a checkout names a path no job wrote, relative to its cwd: it still reads as gone
    (4), but no <name>.lost lands in that checkout, from the CLI or from detach.state."""
    from ttp import detach
    p = make(env)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    r = subprocess.run([sys.executable, str(TTP), "detach", "--check", "job.rc"], cwd=checkout,
                       capture_output=True, text=True, env=dict(os.environ, TTP_PROJECT=str(p.root)))
    assert r.returncode == 4 and "gone" in r.stdout, (r.stdout, r.stderr)
    monkeypatch.chdir(checkout)
    assert detach.state("job.rc")[0] == "lost"
    assert detach._mark_lost_first("other.rc", "x") == "x"
    assert not list(checkout.iterdir()), "a marker landed in the caller's checkout"


def test_lost_markers_go_only_into_state_and_run_folders(env, tmp_path, monkeypatch):
    from ttp import detach
    p = make(env)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setattr(detach.os, "killpg", lambda pid, sig: None)
    detach._mark_lost(outside / "a.rc", "why")
    detach.stop(outside / "b.rc", 4242, "why")
    detach.end_remote(outside / "c.rc", "why")
    assert detach.state(outside / "d.rc")[0] == "lost"
    assert not list(outside.iterdir()), "a marker was written outside the state and run folders"
    run_dir, fix = p.runs / "7", p.state / "heal-fix" / "x"   # a run's folder, the daemon's own jobs
    for d in (run_dir, fix):
        d.mkdir(parents=True)
        detach._mark_lost(d / "job.rc", "killed")
        assert detach.lost(d / "job.rc").read_text().strip() == "killed"
    own = tmp_path / "own-run"   # the calling run's folder, wherever it is
    own.mkdir()
    monkeypatch.setenv("TTP_RUN_DIR", str(own))
    monkeypatch.chdir(own)
    assert detach._mark_lost_first("job.rc", "it never started") == "it never started"
    assert detach.lost(own / "job.rc").read_text().strip() == "it never started"


# `ttp detach --lock <resource>`: the job takes its locks all or nothing through `ttp lock` (cmd_lock)
# before its command starts and holds them for the job's life.

def _lock_job_env(env, p, run_dir, task="41", run="7"):
    return {"TTP_HOME": str(env["home"]), "TTP_HOST": "testhost", "TTP_RUN_DIR": str(run_dir),
            "TTP_TASK": task, "TTP_RUN_ID": run}


def _held_now(p, res):
    from ttp import shared
    return [x for x in shared.held(p) if x.startswith(f"{res}: ")]


def _probe_json(p, run_env, res):
    r = subprocess.run([sys.executable, str(TTP), "lock", "--probe", "--json", res],
                       env={**os.environ, "TTP_PROJECT": str(p.base), **run_env}, capture_output=True, text=True, timeout=60)
    return r.returncode, json.loads(r.stdout)


def test_detach_without_lock_prints_and_records_exactly_as_before(env, tmp_path):
    p = make(env)
    run_dir = p.runs / "7"
    run_dir.mkdir(parents=True)
    r = _ttp_run(p, "detach", "job1", "--", "true", env=_lock_job_env(env, p, run_dir))
    assert r.returncode == 0, r.stderr
    rc = (run_dir / "job1.rc").resolve()
    job = json.loads((run_dir / "detached.json").read_text())[0]
    lines = r.stdout.splitlines()
    assert lines == [f"detached job1 (pid {job['pid']})", f"log: {rc.with_suffix('.log')}", f"rc:  {rc}",
                     f"hand off: status waiting, retry_when \"{TTP} detach --check {rc}\""], lines
    assert job["command"] == "true" and "locks" not in job
    assert _wait_for_path(rc) and rc.read_text().strip() == "0"


def test_detach_lock_holds_one_or_two_locks_for_the_jobs_life_and_lets_go_once_its_rc_is_written(env, tmp_path):
    p = make(env)
    for n, names in enumerate((["rig-a"], ["rig-a", "rig-b"])):
        run_dir = p.runs / str(7 + n)
        run_dir.mkdir(parents=True)
        go, ran = tmp_path / f"go{n}", tmp_path / f"ran{n}"
        flags = [w for x in names for w in ("--lock", x)]
        cmd = ["sh", "-c", f"touch {shlex.quote(str(ran))}; {shlex.join(_until(go))}"]
        run_env = _lock_job_env(env, p, run_dir, run=str(7 + n))
        r = _ttp_run(p, "detach", "job", *flags, "--", *cmd, env=run_env)
        assert r.returncode == 0, r.stderr
        assert f"locks: {','.join(names)} (taken all or nothing" in r.stdout, r.stdout
        rc = (run_dir / "job.rc").resolve()
        job = json.loads((run_dir / "detached.json").read_text())[0]
        assert job["locks"] == names
        try:
            assert _until_true(ran.exists)
            for res in names:   # held by this task's run and job while it runs
                assert _held_now(p, res), res
                code, said = _probe_json(p, run_env, res)
                h = said["holders"][0]
                assert code == 75 and h["holder"] == f"task #41 (run {7 + n})" and h["alive"] is True, said
                assert h["job"] == "job" and h["job_pid"] == job["pid"] and h["job_rc"] == str(rc), h
                assert f"(detached job job, job pid {job['pid']})" in h["note"], h["note"]
            assert _ttp_run(p, "detach", "--check", str(rc)).returncode == 1
        finally:
            go.touch()
        assert _wait_for_path(rc) and rc.read_text().strip() == "0"
        for res in names:
            assert _until_true(lambda: not _held_now(p, res)), res
            assert _probe_json(p, run_env, res)[0] == 0


def test_detach_lock_lets_go_once_the_job_is_lost(env, tmp_path):
    from ttp import detach
    p = make(env)
    run_dir = p.runs / "7"
    run_dir.mkdir(parents=True)
    ran = tmp_path / "ran"
    cmd = ["sh", "-c", f"touch {shlex.quote(str(ran))}; {shlex.join(_until(tmp_path / 'never'))}"]
    r = _ttp_run(p, "detach", "job", "--lock", "rig-a,rig-b", "--", *cmd, env=_lock_job_env(env, p, run_dir))
    assert r.returncode == 0, r.stderr
    rc = (run_dir / "job.rc").resolve()
    pid = json.loads((run_dir / "detached.json").read_text())[0]["pid"]
    assert _until_true(ran.exists) and _held_now(p, "rig-a") and _held_now(p, "rig-b")
    os.killpg(pid, signal.SIGKILL)   # a kill or a reboot: the job is gone without an .rc
    assert _until_true(lambda: detach.state(rc)[0] == "lost"), detach.state(rc)
    assert not rc.exists() and not _held_now(p, "rig-a") and not _held_now(p, "rig-b")


def test_detach_lock_holds_nothing_while_another_task_holds_one_of_them(env, tmp_path):
    from ttp import detach
    p = make(env)
    run_dir = p.runs / "7"
    run_dir.mkdir(parents=True)
    go, ran = tmp_path / "go", tmp_path / "ran"
    other = _lock(_lock_env(env, p, TTP_TASK="99", TTP_RUN_ID="3"), "rig-b", cmd=_until(go))
    try:
        assert _until_true(lambda: bool(_held_now(p, "rig-b")))
        r = _ttp_run(p, "detach", "job", "--lock", "rig-a", "--lock", "rig-b", "--", "touch", str(ran),
                     env=_lock_job_env(env, p, run_dir))
        assert r.returncode == 0, r.stderr
        rc = (run_dir / "job.rc").resolve()
        assert _until_true(lambda: _queue_len(p, "rig-a") == 1 and _queue_len(p, "rig-b") == 1)
        time.sleep(0.5)   # several polls: rig-a is free, yet the job keeps it free while it waits for rig-b
        assert not _held_now(p, "rig-a") and not ran.exists() and detach.state(rc)[0] == "running"
    finally:
        go.touch()
    assert other.wait(timeout=60) == 0
    assert _wait_for_path(rc) and rc.read_text().strip() == "0" and ran.exists()
    assert not _held_now(p, "rig-a") and not _held_now(p, "rig-b")


def test_detach_lock_on_a_paused_resource_exits_75_and_starts_nothing(env, tmp_path):
    p = make(env)
    run_dir = p.runs / "7"
    run_dir.mkdir(parents=True)
    p.db.set_kv("paused_resources", {"rig-b": {"reason": "maintenance", "since": time.time()}})
    r = _ttp_run(p, "detach", "job", "--lock", "rig-a,rig-b", "--", "touch", str(tmp_path / "ran"),
                 env=_lock_job_env(env, p, run_dir))
    assert r.returncode == 75 and "rig-b is paused (maintenance)" in r.stderr, (r.returncode, r.stderr)
    assert "retry_when `ttp lock --probe rig-a,rig-b`" in r.stderr
    assert not (run_dir / "detached.json").exists() and not (run_dir / "job.log").exists()
    assert _queue_len(p, "rig-a") == 0 and not (tmp_path / "ran").exists()
    r = _ttp_run(p, "detach", "--remote", "box", "job", "--lock", "rig-a", "--", "true", env=_lock_job_env(env, p, run_dir))
    assert r.returncode != 0 and "--lock is for a local job" in r.stderr, r.stderr


# A job's detached.json record closes once the job is over (detach.close, the daemon's close_records,
# `ttp detach --reap`). Each was red before the record had a status (no close existed) and is green now.
def _detach_job(p, run_dir, env, name, cmd, *lock):
    r = _ttp_run(p, "detach", name, *[x for res in lock for x in ("--lock", res)], "--", *cmd,
                 env=_lock_job_env(env, p, run_dir))
    assert r.returncode == 0, r.stderr
    return (run_dir / f"{name}.rc").resolve()


def _record(rc):
    return next(j for j in json.loads((rc.parent / "detached.json").read_text()) if j["rc"] == str(rc))


def test_detach_record_of_a_job_that_wrote_its_rc_closes_as_ended_in_one_sweep(env, tmp_path):
    from ttp import detach
    p = make(env)
    run_dir = p.runs / "7"
    run_dir.mkdir(parents=True)
    rc = _detach_job(p, run_dir, env, "job", ["sh", "-c", "exit 3"])
    assert _record(rc)["status"] == "running"
    assert _until_true(lambda: rc.exists() and detach.state(rc)[0] == "ended")
    assert detach.close_records(p) == 1
    e = _record(rc)
    assert (e["status"], e["exit"], e["ended"]) == ("ended", 3, rc.stat().st_mtime)
    assert detach.close_records(p, now=time.time() + 120) == 0 and _record(rc) == e   # closed once


def test_detach_record_of_a_killed_driver_closes_as_died_with_no_exit_code(env, tmp_path):
    from ttp import detach
    p = make(env)
    run_dir = p.runs / "7"
    run_dir.mkdir(parents=True)
    rc = _detach_job(p, run_dir, env, "job", _until(tmp_path / "never"))
    assert _until_true(lambda: detach.started(rc).exists())
    os.killpg(_record(rc)["pid"], signal.SIGKILL)
    assert _until_true(lambda: detach.close(rc) is not None)
    e = _record(rc)
    assert (e["status"], e["exit"]) == ("died", None) and "killed" in e["why"] and not rc.exists()


def test_detach_record_whose_pid_another_process_reuses_closes_and_a_live_one_stays_open(env, tmp_path):
    from ttp import detach
    p = make(env)
    run_dir = p.runs / "7"
    run_dir.mkdir(parents=True)
    gone = _detach_job(p, run_dir, env, "gone", _until(tmp_path / "never"))
    live = _detach_job(p, run_dir, env, "live", _until(tmp_path / "stop"))
    assert _until_true(lambda: detach.started(gone).exists() and detach.started(live).exists())
    os.killpg(_record(gone)["pid"], signal.SIGKILL)
    assert _until_true(lambda: detach.locks.any_free([detach.locks.job_lock(gone)]))
    with detach.registry(run_dir) as jobs:   # its pid now names a live process that is not the job
        next(j for j in jobs if j["name"] == "gone")["pid"] = os.getpid()
    assert detach.close_records(p) == 1
    assert _record(gone)["status"] == "died" and _record(live)["status"] == "running"
    (tmp_path / "stop").touch()
    assert _until_true(lambda: detach.close(live) is not None) and _record(live)["status"] == "ended"


def test_detach_reap_closes_an_ended_job_refuses_a_live_one_and_is_idempotent(env, tmp_path):
    p = make(env)
    run_dir = p.runs / "7"
    run_dir.mkdir(parents=True)
    rc = _detach_job(p, run_dir, env, "job", _until(tmp_path / "stop"))
    e = _lock_job_env(env, p, run_dir)
    r = _ttp_run(p, "detach", "--reap", "job", env=e)
    assert r.returncode == 1 and "still running" in r.stderr and _record(rc)["status"] == "running"
    (tmp_path / "stop").touch()
    assert _until_true(rc.exists)
    first = _ttp_run(p, "detach", "--reap", "job", env=e)
    assert first.returncode == 0 and "job: ended rc=0" in first.stdout, first.stderr
    closed = _record(rc)
    again = _ttp_run(p, "detach", "--reap", str(rc), env=e)
    assert (again.returncode, again.stdout) == (0, first.stdout) and _record(rc) == closed
    assert _ttp_run(p, "detach", "--reap", "other", env=e).returncode == 2
    assert "--reap" in _ttp_run(p, "detach", "--help").stdout


def test_detach_lock_job_closed_as_died_leaves_no_lock_held_or_shown_held(env, tmp_path):
    from ttp import activity, detach
    p = make(env)
    run_dir = p.runs / "7"
    run_dir.mkdir(parents=True)
    rc = _detach_job(p, run_dir, env, "job", _until(tmp_path / "never"), "rig-a,rig-b")
    assert _until_true(lambda: _held_now(p, "rig-a") and _held_now(p, "rig-b"))
    # `ttp lock` logs its takes just after it gets the locks: kill it only once they are logged.
    assert _until_true(lambda: {r["resource"] for r in activity.entries(p.state) if r.get("ev") == "take"}
                       == {"rig-a", "rig-b"})
    os.killpg(_record(rc)["pid"], signal.SIGKILL)   # its `ttp lock` dies too and logs no release
    assert _until_true(lambda: detach.close(rc) is not None) and _record(rc)["status"] == "died"
    assert not _held_now(p, "rig-a") and not _held_now(p, "rig-b")
    recs = activity.entries(p.state)
    takes = [r for r in recs if r.get("ev") == "take"]
    assert {r["resource"] for r in takes} == {"rig-a", "rig-b"} and all(r["job_rc"] == str(rc) for r in takes)
    for r in takes:   # the lock's pid reused by a live process never shows the lock as held
        r["pid"] = os.getpid()
    assert [s["state"] for s in activity.spans(recs, "testhost") if s["kind"] == "lock"] == ["unknown", "unknown"]
