# The daily review's 7-day count per Restrictions item (runtime/ttp/restriction_audit.py). Run from
# test_tt_project_runtime.py in its namespace (see _include_cases there).

_RULES = ("# demo\n\n## Restrictions (binding on every task)\n"
          "- Never reboot the shared build farm while its nightly queue runs.\n"
          "- Never push to the release branch.\n\n## Goals\nShip v1.\n")
_FARM = "Never reboot the shared build farm while its nightly queue runs."


def _blocked_run(p, env, n):
    """A worker run that hands off blocked, through the daemon's hand-off path."""
    from ttp.daemon import Daemon
    from ttp.providers.base import RunUsage as Usage
    tid = p.db.add_task(f"bench {n}", "s", kind="work", tier="light")
    p.db.update_task(tid, status="running")
    run_dir = env["tmp"] / f"run-rule-{tid}-{time.monotonic_ns()}"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(json.dumps(
        {"status": "blocked", "summary": "the next step reboots the shared build farm",
         "question": "the charter bans a reboot of the shared build farm while its nightly queue runs: may I?"}))
    d = Daemon(p.base)
    with d.p.db.tx():
        d._finish_worker({"task": tid}, Usage(cost_usd=0.1), "ok", run_dir)
    assert p.db.task(tid)["status"] == "blocked"


def _refused(p, text):
    """A coordinator action the charter guard refuses, recorded as the daemon records a turn's rejections."""
    from ttp import coordinator as coord
    from ttp.daemon import Daemon
    problems = coord.apply(p, [{"type": "charter_update", "section": "Policies", "text": text}], turn=1)
    assert problems and "contradicts the Restrictions item" in problems[0], problems
    Daemon(p.base)._record_rejections(problems)


def _audit(p):
    from ttp import audit
    return [x for x in audit.lines(p.db, p=p) if x.startswith("Restrictions item only blocked work")]


def test_a_restriction_cited_by_three_blocked_runs_and_no_refusal_is_listed(env):
    from ttp import restriction_audit
    p = make(env)
    p.charter_path.write_text(_RULES)
    for n in range(3):
        _blocked_run(p, env, n)
    (line,) = _audit(p)
    assert "held 3 blocked runs and 0 asks, stopped no banned action" in line and _FARM in line, line
    assert "release branch" not in line
    farm, release = restriction_audit.citations(p)
    assert (farm["blocked"], farm["stopped"], release["blocked"]) == (3, 0, 0)


def test_asks_count_as_blocked_work_and_the_threshold_is_the_config_key(env):
    from ttp import restriction_audit
    p = make(env)
    p.charter_path.write_text(_RULES)
    _blocked_run(p, env, 0)
    problems, ask = _ask(p, "The nightly queue runs on the shared build farm: may I reboot it now, as the charter bans?",
                         blocking="restriction", classify="restriction_change",
                         least_disruptive="wait for the queue to end; it breaks the build farm restriction",
                         recommendation="wait for the queue to end")
    assert problems == [] and ask, problems
    assert restriction_audit.citations(p)[0]["asks"] == 1
    assert _audit(p) == []   # 2 under the default 3
    cfg = json.loads(p.config_path.read_text())
    cfg.setdefault("review", {})["rule_blocks_min"] = 2
    p.config_path.write_text(json.dumps(cfg))
    (line,) = _audit(p)
    assert "held 1 blocked run and 1 ask" in line, line


def test_a_restriction_that_also_stopped_a_banned_action_is_not_listed(env):
    from ttp import restriction_audit
    p = make(env)
    p.charter_path.write_text(_RULES)
    for n in range(3):
        _blocked_run(p, env, n)
    _refused(p, "Workers may reboot the shared build farm while its nightly queue runs.")
    assert restriction_audit.citations(p)[0]["stopped"] == 1
    assert _audit(p) == []


def test_a_restriction_under_the_threshold_is_not_listed(env):
    from ttp import restriction_audit
    p = make(env)
    p.charter_path.write_text(_RULES)
    for n in range(2):
        _blocked_run(p, env, n)
    assert restriction_audit.citations(p)[0]["blocked"] == 2
    assert _audit(p) == []
    # Older than 7 days does not count.
    for n in range(2, 4):
        _blocked_run(p, env, n)
    assert restriction_audit.citations(p, now=time.time() + 8 * 86400)[0]["blocked"] == 0


def test_a_charter_with_no_restrictions_gives_no_line(env):
    from ttp import restriction_audit
    p = make(env)
    p.charter_path.write_text("# demo\n\n## Restrictions (binding on every task)\n(none stated yet)\n\n## Goals\n"
                              "Never reboot the shared build farm while its nightly queue runs.\n")
    for n in range(3):
        _blocked_run(p, env, n)
    assert restriction_audit.citations(p) == [] and _audit(p) == []
