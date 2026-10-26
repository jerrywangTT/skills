# Parallel code tasks naming the same files (overlap.py): a warning at task_add, in the digest's
# open tasks and in the worker prompt, never a refusal. Run from tests/test_tt_project_runtime.py
# (see _include_cases there): helpers, fixtures and imports come from that file.
from __future__ import annotations


def _branch_adding(p, branch, files):
    """A branch off the root's checked-out base with one commit adding (or changing) `files`."""
    def git(*args):
        return subprocess.run(["git", "-C", str(p.root), *args], check=True, capture_output=True,
                              text=True).stdout.strip()
    home = git("symbolic-ref", "--short", "HEAD")
    git("checkout", "-q", "-b", branch)
    for f, text in files.items():
        (p.root / f).parent.mkdir(parents=True, exist_ok=True)
        (p.root / f).write_text(text)
        git("add", f)
    git("commit", "-q", "-m", f"work on {branch}")
    head = git("rev-parse", "HEAD")
    git("checkout", "-q", home)
    return head


def test_task_add_warns_when_another_open_code_task_names_the_same_new_file(env):
    """#3466 and #3467 each built their own runtime/ttp/progress.py side by side. A new code task
    whose spec names a file an open code task's spec names is still added, and its result and its
    open-tasks line in the digest name the other task, the shared path and the fix; once it waits on
    that task the digest line drops the warning."""
    from ttp import coordinator as coord
    p = make(env)
    first = p.db.add_task("progress check", "Add runtime/ttp/progress.py and tests/tt_project_runtime/"
                          "progress_cases.py with a stall detector.", kind="code", status="running")
    assert coord.apply(p, [{"type": "task_add", "title": "driver reports", "kind": "code",
                            "spec": "Record driver results in `runtime/ttp/progress.py` (append, KINDS)."}]) == []
    new = p.db.one("SELECT id FROM tasks WHERE title='driver reports'")["id"]
    want = f"overlaps #{first} (runtime/ttp/progress.py): build on it, not beside it: depends_on [{first}]"
    assert f"task_add: #{new} added; {want}" in p.db.kv(coord.NOTES_KEY)
    assert _labels(p, new) == [f"overlap:{first}:{want}"]
    line = next(x for x in coord.digest(p, {}, [], []).splitlines() if x.startswith(f"- #{new} |"))
    assert line.endswith(f"| driver reports | {want}"), line
    p.db.update_task(new, depends_on=json.dumps([first]))
    line = next(x for x in coord.digest(p, {}, [], []).splitlines() if x.startswith(f"- #{new} |"))
    assert "overlaps" not in line, line
    # A plan or question task, or a code task that names other files, gets no warning.
    for title, kind, spec in (("plan it", "plan", "Plan runtime/ttp/progress.py."),
                              ("other file", "code", "Change runtime/ttp/relay.py only.")):
        assert coord.apply(p, [{"type": "task_add", "title": title, "kind": kind, "spec": spec}]) == []
        tid = p.db.one("SELECT id FROM tasks WHERE title=?", (title,))["id"]
        assert not [x for x in _labels(p, tid) if x.startswith("overlap:")], title
    assert not [n for n in p.db.kv(coord.NOTES_KEY) if "overlaps" in n and "driver reports" not in n
                and f"#{new} " not in n]


def test_task_add_flags_a_file_an_unlanded_finished_branch_adds_as_a_new_file(env):
    """A done code task's branch that has not landed adds runtime/ttp/progress.py: a new task naming
    that file is warned with the file flagged as new; one that names a file the branch only changes
    is warned without the flag. Once the records show the task landed, nothing is flagged."""
    from ttp import coordinator as coord, overlap
    p = make(env)
    p.set_config("delivery.landed_probe", "test -e landed/{id}")   # no push branch: how landed:#<id> is checked
    (p.root / "runtime" / "ttp").mkdir(parents=True, exist_ok=True)
    (p.root / "runtime/ttp/relay.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(p.root), "add", "runtime/ttp/relay.py"], check=True)
    subprocess.run(["git", "-C", str(p.root), "commit", "-q", "-m", "relay"], check=True)
    done = p.db.add_task("progress check", "s", kind="code", status="done")
    branch = f"ttp/t{done}-progress-check"
    head = _branch_adding(p, branch, {"runtime/ttp/progress.py": "def append(state, rec): ...\n",
                                      "runtime/ttp/relay.py": "x = 2\n"})
    p.db.update_task(done, branch=branch)
    assert coord.apply(p, [{"type": "task_add", "title": "stall watcher", "kind": "code",
                            "spec": "Extend plugins/tt-project/runtime/ttp/progress.py with a stall streak."}]) == []
    new = p.db.one("SELECT id FROM tasks WHERE title='stall watcher'")["id"]
    note = next(n for n in p.db.kv(coord.NOTES_KEY) if n.startswith(f"task_add: #{new} added; "))
    assert f"overlaps #{done} (plugins/tt-project/runtime/ttp/progress.py (new file))" in note, note
    assert f"start_when landed:#{done}" in note
    assert coord.apply(p, [{"type": "task_add", "title": "relay tweak", "kind": "code",
                            "spec": "Tidy runtime/ttp/relay.py."}]) == []
    tweak = p.db.one("SELECT id FROM tasks WHERE title='relay tweak'")["id"]
    assert overlap.digest_note(p.db.task(tweak)).startswith(f"overlaps #{done} (runtime/ttp/relay.py): "), \
        _labels(p, tweak)
    # The diff is read once per head and kept in state/.
    assert json.loads((p.state / overlap.CACHE).read_text())[head] == [["A", "runtime/ttp/progress.py"],
                                                                       ["M", "runtime/ttp/relay.py"]]
    # Landed (the push queue recorded it): no longer an overlap.
    p.db.x("INSERT INTO task_landings(marker, task, sha, target) VALUES('m1', ?, ?, 'main')", (done, head))
    assert [h["id"] for h in overlap.find(p, "Extend runtime/ttp/progress.py")] == [new]   # its spec, not the branch


def test_task_add_gives_no_warning_for_ignored_common_files_hubs_or_bare_names(env):
    """Files most changes touch never warn: changelogs, changesets, manifests and __init__.py (the
    overlap.ignore globs), hubs (tracked files over overlap.hub_bytes), and file names without a
    folder (named in passing). URLs and words like and/or are no paths."""
    from ttp import coordinator as coord, overlap
    p = make(env)
    (p.root / "runtime" / "ttp").mkdir(parents=True, exist_ok=True)
    (p.root / "runtime/ttp/daemon.py").write_text("#" * 2000 + "\n")
    subprocess.run(["git", "-C", str(p.root), "add", "runtime/ttp/daemon.py"], check=True)
    subprocess.run(["git", "-C", str(p.root), "commit", "-q", "-m", "hub"], check=True)
    p.set_config("overlap.hub_bytes", 1000)
    common = ("Bump .claude-plugin/plugin.json, runtime/ttp/__init__.py and CHANGELOG.md, add "
              ".changeset/x.md, wire runtime/ttp/daemon.py; see progress.py, and/or "
              "https://example.com/runtime/ttp/progress.py.")
    other = p.db.add_task("first", common, kind="code", status="queued")
    assert overlap.spec_paths(common, p.root) == ["runtime/ttp/__init__.py", "runtime/ttp/daemon.py"]
    assert overlap.keep(p, overlap.spec_paths(common, p.root)) == []
    assert coord.apply(p, [{"type": "task_add", "title": "second", "kind": "code", "spec": common}]) == []
    new = p.db.one("SELECT id FROM tasks WHERE title='second'")["id"]
    assert not [x for x in _labels(p, new) if x.startswith("overlap:")]
    assert not [n for n in p.db.kv(coord.NOTES_KEY) if "overlaps" in n]
    assert overlap.find(p, "Rename runtime/ttp/other.py") == [] and other
    # Turned off: no warning even for a real overlap.
    p.db.add_task("third", "Add runtime/ttp/new_mod.py", kind="code", status="queued")
    assert overlap.find(p, "Use runtime/ttp/new_mod.py")
    p.set_config("overlap.enabled", False)
    assert overlap.find(p, "Use runtime/ttp/new_mod.py") == []


def test_worker_prompt_lists_unlanded_branches_that_touch_its_files(env):
    """A code worker's prompt lists the other unlanded branches (task, branch, head) that add or change
    a path its spec names, telling it to reuse their module; its own branch and unrelated or landed
    branches are not listed."""
    from ttp.prompts import worker_task
    p = make(env)
    done = p.db.add_task("progress check", "s", kind="code", status="done")
    branch = f"ttp/t{done}-progress-check"
    head = _branch_adding(p, branch, {"runtime/ttp/progress.py": "def append(state, rec): ...\n"})
    p.db.update_task(done, branch=branch)
    elsewhere = p.db.add_task("docs", "s", kind="code", status="done")
    _branch_adding(p, f"ttp/t{elsewhere}-docs", {"docs/guide.md": "x\n"})
    p.db.update_task(elsewhere, branch=f"ttp/t{elsewhere}-docs")
    tid = p.db.add_task("stall watcher", "Add a stall streak to runtime/ttp/progress.py.", kind="code")
    prompt = worker_task(p, p.db.task(tid), str(p.root), None)
    block = prompt[prompt.index("## Unlanded branches that touch your files"):]
    assert "reuse their module and API" in block and "never create a second one" in block
    assert f"- #{done} {branch} at {head[:10]}: runtime/ttp/progress.py (new file)" in block, block
    assert f"#{elsewhere}" not in block
    plain = p.db.add_task("relay", "Change runtime/ttp/relay.py.", kind="code")
    assert "## Unlanded branches" not in worker_task(p, p.db.task(plain), str(p.root), None)


def test_task_add_gives_no_warning_for_folders_or_the_required_check_files(env):
    """Nearly every spec names its folder ('in plugins/tt-project (runtime/ttp)') and the required
    suite: neither is an overlap. Folders the repo has are no paths, and the files the check commands
    (delivery.push_checks) name never count, however small; a shared runtime/ttp/progress.py still warns."""
    from ttp import coordinator as coord, overlap
    p = make(env)
    files = ["plugins/tt-project/runtime/ttp/relay.py", "plugins/tt-project/runtime/ttp/webapp.py",
             "tests/test_plugin_catalogue.py", "tests/test_tt_project_plugin.py",
             "tests/test_tt_project_runtime.py", "tests/tt_project_runtime/relay_cases.py"]
    for f in files:
        (p.root / f).parent.mkdir(parents=True, exist_ok=True)
        (p.root / f).write_text("x = 1\n")
    subprocess.run(["git", "-C", str(p.root), "add", *files], check=True)
    subprocess.run(["git", "-C", str(p.root), "commit", "-q", "-m", "tree"], check=True)
    overlap._ls_cache.clear()
    suite = ("python3 -m pytest -q tests/test_tt_project_plugin.py tests/test_tt_project_runtime.py "
             "tests/test_plugin_catalogue.py")
    p.set_config("delivery.push_checks", [suite, "claude plugin validate plugins/tt-project --strict"])
    accept = f"Acceptance: `{suite}` green; cases in tests/tt_project_runtime."
    relay = f"In plugins/tt-project (runtime/ttp): fix the relay's retry. {accept}"
    web = f"In plugins/tt-project (runtime/ttp): show spend in the web app. {accept}"
    assert overlap.spec_paths(relay, p.root) == ["tests/test_tt_project_plugin.py", "tests/test_tt_project_runtime.py",
                                                 "tests/test_plugin_catalogue.py"]
    assert overlap.keep(p, overlap.spec_paths(relay, p.root)) == []
    first = p.db.add_task("relay retry", relay, kind="code", status="running")
    assert coord.apply(p, [{"type": "task_add", "title": "web spend", "kind": "code", "spec": web}]) == []
    new = p.db.one("SELECT id FROM tasks WHERE title='web spend'")["id"]
    assert not [x for x in _labels(p, new) if x.startswith("overlap:")]
    assert not [n for n in p.db.kv(coord.NOTES_KEY) if "overlaps" in n]
    # A shared file still warns, with the folder and suite left out of the warning.
    p.db.update_task(first, spec=relay + " Add runtime/ttp/progress.py.")
    hits = overlap.find(p, web + " Use runtime/ttp/progress.py.")
    assert [(h["id"], h["paths"]) for h in hits] == [(first, [("runtime/ttp/progress.py", False)])]


def test_overlap_lists_only_open_or_recent_unlanded_tasks_with_advice_each_can_follow(env):
    """#3526's warning listed #1115 and 27 more tasks done two weeks earlier whose work had long landed
    through a successor (a review, a fix) or was shelved, telling it to wait on `landed:#1115`. Only
    an open task (depends_on) or a done one that finished within overlap.branch_days with a head
    still off the push branch (start_when landed:#<id>) counts; the most recent are listed first."""
    from ttp import coordinator as coord, overlap
    p = make(env)
    root = p.root
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                              text=True).stdout.strip()
    p.set_config("delivery.push_branch", git("symbolic-ref", "--short", "HEAD"))
    path = "runtime/ttp/progress.py"
    now = time.time()

    def done_task(title, ended, **kw):
        tid = p.db.add_task(title, "s", kind="code", status="done", **kw)
        _branch_adding(p, f"ttp/t{tid}-x", {path: f"# {title}\n"})
        p.db.update_task(tid, branch=f"ttp/t{tid}-x")
        p.db.x("INSERT INTO runs(task, role, status, started, ended) VALUES(?,'worker','done',?,?)",
               (tid, ended - 60, ended))
        return tid
    old = done_task("old, no landed record", now - 20 * 86400)          # a later write moved `updated`
    carried = done_task("landed through its fix", now - 3600)
    fix = p.db.add_task("Fix review of it", "s", kind="code", status="done", labels=[f"review_fix:{carried}"])
    p.db.x("INSERT INTO task_landings(marker, task, sha, target) VALUES('m1', ?, 'abc1234', 'b')", (fix,))
    shelved = done_task("shelved", now - 3600)
    p.db.update_task(shelved, result=json.dumps({"summary": "no gain", "outcome": "shelved"}))
    picked = done_task("on the push branch by patch", now - 3600)
    recent = done_task("recent, unlanded", now - 3600)
    git("commit", "-q", "--allow-empty", "-m", "other work")   # so the pick is a new commit, not the same one
    git("cherry-pick", f"ttp/t{picked}-x")
    opened = p.db.add_task("open one", f"Add {path}.", kind="code", status="queued")
    hits = overlap.find(p, f"Extend {path}.")
    assert [(h["id"], h["open"]) for h in hits] == [(opened, True), (recent, False)], (hits, old, carried, shelved)
    want = (f"overlaps #{opened} ({path}); #{recent} ({path} (new file)): build on them, not beside them: "
            f"depends_on [{opened}] and start_when landed:#{recent}")
    assert overlap.warning(hits) == want
    assert overlap.label(hits) == f"overlap:{opened},{recent}:{want}"
    assert json.loads((p.state / overlap.LANDED_CACHE).read_text()) == [git("rev-parse", f"ttp/t{picked}-x")]
    # Only the done one: a landed wait it can pass; once it lands, nothing.
    p.db.update_task(opened, status="cancelled")
    assert overlap.warning(overlap.find(p, f"Extend {path}.")) == (
        f"overlaps #{recent} ({path} (new file)): build on it, not beside it: start_when landed:#{recent}")
    p.db.x("INSERT INTO task_landings(marker, task, sha, target) VALUES('m2', ?, 'def5678', 'b')", (recent,))
    assert overlap.find(p, f"Extend {path}.") == []
    assert coord.apply(p, [{"type": "task_add", "title": "stall", "kind": "code", "spec": f"Extend {path}."}]) == []
    assert not [n for n in p.db.kv(coord.NOTES_KEY) if "overlaps" in n]


def test_overlap_warning_names_the_most_recent_few_then_a_count(env):
    """Many matches: the warning names the MAX_TASKS most recent, then '+N more tasks', and its fix
    and label name only those; the digest drops it once the task depends on them."""
    from ttp import coordinator as coord, overlap
    p = make(env)
    ids = [p.db.add_task(f"t{i}", "Change runtime/ttp/relay.py.", kind="code", status="queued") for i in range(5)]
    assert coord.apply(p, [{"type": "task_add", "title": "mine", "kind": "code",
                            "spec": "Tidy runtime/ttp/relay.py."}]) == []
    new = p.db.one("SELECT id FROM tasks WHERE title='mine'")["id"]
    top = ids[::-1][:overlap.MAX_TASKS]
    want = ("overlaps " + "; ".join(f"#{i} (runtime/ttp/relay.py)" for i in top) + "; +2 more tasks: build on "
            f"them, not beside them: depends_on [{', '.join(map(str, top))}]")
    assert f"task_add: #{new} added; {want}" in p.db.kv(coord.NOTES_KEY)
    assert _labels(p, new) == [f"overlap:{','.join(map(str, top))}:{want}"]
    p.db.update_task(new, depends_on=json.dumps(top))
    assert overlap.digest_note(p.db.task(new)) == ""


def _failed_review(p, of, ago=3600):
    """A review of task `of` that failed `ago` seconds back (a task_failed event on it)."""
    rid = p.db.add_task(f"Review #{of}", "s", kind="review", status="failed", depends_on=[of])
    p.db.x("INSERT INTO events(ts, source, kind, text, task) VALUES(?, 'daemon', 'task_failed', 'x', ?)",
           (time.time() - ago, rid))
    return rid


def test_task_add_chains_a_new_task_on_a_hot_or_settled_file_after_the_newest_open_task(env, monkeypatch):
    """Five lineages changed runtime/ttp/progress.py side by side and broke each other's reviews. A new
    code task naming a file a review failed on within 48 h (its main file, change not landed) waits
    for the newest open task sharing it to land (start_when landed:#<id>, ANDed with its own wait);
    a `parallel: <why>` line skips that and is recorded; a file with no failed review keeps only the
    warning; a landed or cancelled open task chains nothing; a settled area's file chains too."""
    from ttp import coordinator as coord, overlap, reviewcap, settled
    from ttp.db import deferral
    p = make(env)
    p.set_config("delivery.landed_probe", "test -e landed/{id}")   # no push branch: how landed:#<id> is checked
    hot, cold, calm = "runtime/ttp/progress.py", "runtime/ttp/relay.py", "runtime/ttp/sweep.py"
    mains = {}
    monkeypatch.setattr(reviewcap, "main_file", lambda p, r: mains.get(r["id"]))

    def add(title, spec, **kw):
        assert coord.apply(p, [{"type": "task_add", "title": title, "kind": "code", "spec": spec, **kw}]) == []
        return p.db.one("SELECT * FROM tasks WHERE title=?", (title,))

    older = p.db.add_task("older", f"Change {hot}.", kind="code", status="done")
    mains[_failed_review(p, older)] = hot
    first = p.db.add_task("first", f"Change {hot}.", kind="code", status="queued")
    newest = p.db.add_task("newest", f"Change {hot} and {cold}.", kind="code", status="running")
    t = add("chained", f"Extend {hot}.")
    assert deferral(t)["when"] == f"landed:#{newest}", deferral(t)
    assert f"#{newest} lands" in deferral(t)["why"]
    assert any(n.startswith(f"task_add: #{t['id']} chained after #{newest} (start_when landed:#{newest}): {hot}")
               for n in p.db.kv(coord.NOTES_KEY)), p.db.kv(coord.NOTES_KEY)
    assert overlap.digest_note(t) == ""   # waiting on the newest settles the warning
    # The next one chains after that one (now the newest), the caller's own wait kept as an AND.
    prev = t["id"]
    t = add("own wait", f"Extend {hot}.", start_when=f"landed:#{first}")
    assert deferral(t)["when"] == f"ttp landed --task {prev} && ttp landed --task {first}"
    prev = t["id"]
    t = add("own probe", f"Extend {hot}.", start_when="test -e /tmp/x || false")
    assert deferral(t)["when"] == f"ttp landed --task {prev} && (test -e /tmp/x || false)"
    # An ancestor probe keeps landed.probe_command's rewrite, so a sha the push queue rebased still counts.
    # "own probe" waits on a probe of its own (overlap.exempt): it is passed over for the one before it.
    sha = "abcdef1234567"
    t = add("rebased sha check", f"Extend {hot} for the queue.", start_when=f"git merge-base --is-ancestor {sha} origin/tt/push")
    assert deferral(t)["when"] == f"ttp landed --task {prev} && ttp landed {sha} --onto origin/tt/push"
    # A combined probe over the start_when limit chains nothing: the caller's own wait stays.
    long = "test -e /tmp/" + "x" * (coord.START_WHEN_CHARS - 30)
    t = add("oversized shell gate", f"Guard {hot} with a file test.", start_when=long)
    assert deferral(t)["when"] == long
    # A `parallel:` line skips the chain and is recorded on the task.
    t = add("beside", f"Extend {hot}.\nparallel: a one-line hotfix the other task does not touch")
    assert not deferral(t)
    assert "parallel:a one-line hotfix the other task does not touch" in json.loads(t["labels"])
    assert any(f"#{t['id']} runs beside" in n for n in p.db.kv(coord.NOTES_KEY))
    # A file with no failed review: the warning label only.
    t = add("cold", f"Extend {cold}.")
    assert not deferral(t) and [x for x in json.loads(t["labels"]) if x.startswith("overlap:")]
    # Once the failed review's change landed, the area is no longer hot.
    p.db.x("INSERT INTO task_landings(marker, task, sha, target) VALUES('m1', ?, 'abc1234', 'b')", (older,))
    assert not deferral(add("after landing", f"Extend {hot}."))
    # A landed or cancelled open task chains nothing, while the area stays hot.
    mains[_failed_review(p, p.db.add_task("stray", f"Change {hot}.", kind="code", status="done"))] = hot
    p.db.x("UPDATE tasks SET status='cancelled' WHERE kind='code' AND status='queued' AND id>?", (newest,))
    assert overlap.chain(p, overlap.find(p, f"Extend {hot}."))[0] == newest
    p.db.update_task(newest, status="cancelled")
    p.db.x("INSERT INTO task_landings(marker, task, sha, target) VALUES('m2', ?, 'def5678', 'b')", (first,))
    assert [h["id"] for h in overlap.find(p, f"Extend {hot}.")] == [first]
    assert overlap.chain(p, overlap.find(p, f"Extend {hot}.")) is None
    # A settled area's file chains after the newest open task touching it, failed review or not.
    p.db.update_task(newest, status="queued")
    settled.add(p, [calm], "one rule", commit=False)
    p.db.update_task(first, spec=f"Change {calm}.")
    later = p.db.add_task("later", f"Change {calm}.", kind="code", status="queued")
    t = add("settled", f"Extend {calm}.")
    assert deferral(t)["when"] == f"landed:#{later}"
    assert f"{calm} is a settled area" in deferral(t)["why"]


def test_overlap_chain_never_waits_on_a_task_that_waits_on_the_new_tasks_own_lineage(env, monkeypatch):
    """A failed review's code task A, then B chained after it (start_when landed:#A): the fix of A
    (continues A) must not wait on B, as B waits on A, whose work lands only with the fix. Two hops
    (C waits on B, which waits on A) are followed too; an unrelated open task is still chained."""
    from ttp import coordinator as coord, reviewcap
    from ttp.db import deferral
    p = make(env)
    p.set_config("delivery.landed_probe", "test -e landed/{id}")   # no push branch: how landed:#<id> is checked
    hot = "runtime/ttp/progress.py"
    monkeypatch.setattr(reviewcap, "main_file", lambda p, r: hot)

    def add(title, spec, **kw):
        assert coord.apply(p, [{"type": "task_add", "title": title, "kind": "code", "spec": spec, **kw}]) == []
        return p.db.one("SELECT * FROM tasks WHERE title=?", (title,))

    a = p.db.add_task("A", f"Change {hot}.", kind="code", status="done")
    _failed_review(p, a)
    b = p.db.add_task("B", f"Change {hot}.", kind="code", status="queued",
                      labels=[f"start_when:landed:#{a}"])
    fix = add("Fix A", f"Fix {hot}.", continues=a)
    assert not deferral(fix), deferral(fix)
    p.db.update_task(fix["id"], status="cancelled")
    c = p.db.add_task("C", f"Change {hot}.", kind="code", status="queued", depends_on=[b])
    fix = add("Fix A again", f"Fix {hot}.", continues=a)
    assert not deferral(fix), deferral(fix)
    p.db.update_task(fix["id"], status="cancelled")
    # Only C and B wait on A: an unrelated open task newer than them is still the one chained after.
    d = p.db.add_task("D", f"Change {hot}.", kind="code", status="queued")
    fix = add("Fix A third", f"Fix {hot}.", continues=a)
    assert deferral(fix)["when"] == f"landed:#{d}"
    # A task outside A's lineage never chains after C or B either: each waits itself (overlap.exempt).
    p.db.x("UPDATE tasks SET status='cancelled' WHERE id IN (?, ?)", (d, fix["id"]))
    assert not deferral(add("other", f"Extend {hot}."))


def test_an_automatic_chain_uses_depends_on_where_no_landing_can_be_checked(env, monkeypatch):
    """A project with no delivery.push_branch lands its work some other way, so `landed:#<id>` would
    exit 2 on every check. With no delivery.landed_probe either, task_add's automatic chains (a
    settled area's file, a file a review failed on within 48 h) wait on the open task with depends_on,
    and the overlap warning suggests no landed wait. With delivery.landed_probe set they keep
    `landed:#<id>`, as does a project with a push branch."""
    from ttp import coordinator as coord, overlap, reviewcap, settled
    from ttp.db import deferral, dependency_ids
    p = make(env)
    hot, calm = "runtime/ttp/progress.py", "runtime/ttp/sweep.py"
    mains = {}
    monkeypatch.setattr(reviewcap, "main_file", lambda p, r: mains.get(r["id"]))

    def add(title, spec, **kw):
        assert coord.apply(p, [{"type": "task_add", "title": title, "kind": "code", "spec": spec, **kw}]) == []
        return p.db.one("SELECT * FROM tasks WHERE title=?", (title,))

    older = p.db.add_task("older", f"Change {hot}.", kind="code", status="done")
    mains[_failed_review(p, older)] = hot
    open_ = p.db.add_task("open", f"Change {hot}.", kind="code", status="running")
    t = add("failed review file", f"Extend {hot}.")
    assert not deferral(t) and dependency_ids(t) == [open_], (deferral(t), t["depends_on"])
    assert any(n.startswith(f"task_add: #{t['id']} chained after #{open_} (depends_on [{open_}]") and
               "landed:#" not in n for n in p.db.kv(coord.NOTES_KEY)), p.db.kv(coord.NOTES_KEY)
    assert overlap.digest_note(t) == ""   # it waits on the task its label names
    settled.add(p, [calm], "one rule", commit=False)
    other = p.db.add_task("settled work", f"Change {calm}.", kind="code", status="queued")
    t = add("settled area", f"Extend {calm}.")
    assert not deferral(t) and dependency_ids(t) == [other]
    # A done task with a head off the branch: no `start_when landed:#<id>` advice here.
    hits = [{"id": 7, "open": False, "head": "abc1234", "paths": [(hot, False)]}]
    assert "landed:#" not in overlap.warning(hits, landing=False) and "landed:#7" in overlap.warning(hits)
    # With a landed probe, or a push branch, the chain is a landed wait as before.
    p.set_config("delivery.landed_probe", "test -e landed/{id}")
    t = add("with a landed probe", f"Extend {calm}.")
    assert deferral(t)["when"] == f"landed:#{t['id'] - 1}" and dependency_ids(t) == []
    p.set_config("delivery.landed_probe", "")
    p.set_config("delivery.push_branch", "tt/push")
    t = add("with a push branch", f"Extend {calm}.")
    assert deferral(t)["when"] == f"landed:#{t['id'] - 1}" and dependency_ids(t) == []


def test_overlap_never_chains_or_suggests_depends_on_onto_device_or_waiting_tasks(env, monkeypatch):
    """A delivery sat behind a long device A/B run and a fix that itself waited on another task,
    because an overlap became depends_on. A new task sharing a hot file with an open task that holds
    or waits for a device or machine lock, or that is blocked, deferred or waits on unfinished
    depends_on, gets no depends_on or chain on it: the overlap stays a label and the warning says
    why. An ordinary open code task is still chained after, as before."""
    from ttp import coordinator as coord, overlap, reviewcap
    from ttp.db import deferral, dependency_ids
    p = make(env)
    p.set_config("device.locks", ["board0"])
    hot = "runtime/ttp/progress.py"
    monkeypatch.setattr(reviewcap, "main_file", lambda p, r: hot)

    def add(title, spec, **kw):
        assert coord.apply(p, [{"type": "task_add", "title": title, "kind": "code", "spec": spec, **kw}]) == []
        return p.db.one("SELECT * FROM tasks WHERE title=?", (title,))

    _failed_review(p, p.db.add_task("older", f"Change {hot}.", kind="code", status="done"))
    ordinary = p.db.add_task("ordinary", f"Change {hot}.", kind="code", status="queued")
    other = p.db.add_task("unrelated", "Write the notes.", kind="work", status="queued")
    cases = [("device run", "running", {"labels": ["exclusive:box-device"]}, "holds or waits for box-device"),
             ("board job", "queued", {"labels": ["resource:board0"]}, "holds or waits for board0"),
             ("blocked", "blocked", {}, "is blocked"),
             ("deferred", "queued", {"labels": ["start_when:test -e /tmp/never"]}, "is deferred"),
             ("waits", "queued", {"depends_on": [other]}, f"waits on #{other}")]
    for i, (name, status, kw, why) in enumerate(cases):
        h = p.db.add_task(name, f"Change {hot}.", kind="code", status=status, **kw)
        t = add(f"new {i}", f"Extend {hot}.")
        # No landing to check here: an automatic chain would be depends_on.
        assert dependency_ids(t) == [ordinary] and not deferral(t), (i, t["depends_on"], deferral(t))
        assert overlap.exempt(p, p.db.task(h)) == why
        lb = [x for x in json.loads(t["labels"]) if x.startswith("overlap:")]
        assert lb and f"#{h} (" in lb[0] and lb[0].startswith(f"overlap:{ordinary}:"), lb
        note = next(n for n in p.db.kv(coord.NOTES_KEY) if n.startswith(f"task_add: #{t['id']} added; overlaps"))
        assert f"no depends_on on #{h} {why}" in note and f"depends_on [{h}" not in note, note
        p.db.update_task(t["id"], status="cancelled")
        p.db.update_task(h, status="cancelled")
    held = p.db.one("SELECT id FROM tasks WHERE title='device run'")["id"]
    # With a landing to wait on, the chain is landed:#<ordinary>, never a device task's landing.
    p.set_config("delivery.landed_probe", "test -e landed/{id}")
    p.db.update_task(held, status="running")
    t = add("with a landed probe", f"Extend {hot}.")
    assert deferral(t)["when"] == f"landed:#{ordinary}" and dependency_ids(t) == []


def test_a_continue_hop_drops_overlap_depends_on_onto_device_or_waiting_tasks(env):
    """A continue hop was queued with depends_on [A, B] copied from its predecessor's overlap
    warning: A a long device run, B a fix itself waiting on another task. The hop drops edges its
    predecessor's overlap label names onto such tasks and says so; an overlap-named edge onto an
    ordinary task and an edge the coordinator set on purpose (not in the label) are carried."""
    from ttp import coordinator as coord
    from ttp.db import dependency_ids
    p = make(env)
    f = "runtime/ttp/progress.py"
    later = p.db.add_task("later", "Land the next batch.", kind="work", status="queued")
    a = p.db.add_task("A/B device run", f"Change {f}.", kind="code", status="running",
                      labels=["exclusive:box-device"])
    b = p.db.add_task("tests-only fix", f"Change {f}.", kind="code", status="queued", depends_on=[later])
    plain = p.db.add_task("plain", f"Change {f}.", kind="code", status="queued")
    mine = p.db.add_task("explicit", "Prepare the data.", kind="work", status="queued",
                         labels=["exclusive:box-device"])
    text = f"overlaps #{plain} ({f}); #{b} ({f}); #{a} ({f}): build on them, not beside them"
    for status in ("failed", "done"):
        old = p.db.add_task(f"delivery {status}", "Deliver it.", kind="code", status=status,
                            depends_on=[a, b, plain, mine], labels=[f"overlap:{plain},{b},{a}:{text}"])
        title = f"continue: delivery {status}"
        assert coord.apply(p, [{"type": "task_add", "title": title, "kind": "code", "continues": old,
                                "spec": "Finish the delivery.", "depends_on": [a, b, plain, mine]}]) == []
        t = p.db.one("SELECT * FROM tasks WHERE title=?", (title,))
        assert dependency_ids(t) == [plain, mine], t["depends_on"]
        note = next(n for n in p.db.kv(coord.NOTES_KEY) if n.startswith(f"task_add: #{t['id']} continues #{old}"))
        assert f"#{a} (holds or waits for box-device)" in note and f"#{b} (waits on #{later})" in note, note
        p.db.update_task(t["id"], status="cancelled")
    # The coordinator's prompt states the exemption.
    rule = " ".join((RUNTIME.parent / "template" / "prompts" / "coordinator.md").read_text().split())
    assert "Never make an overlap a depends_on onto a device or machine task or one that waits itself" in rule
    # A plain task_add (no hop) keeps the depends_on the coordinator gave, device task or not.
    assert coord.apply(p, [{"type": "task_add", "title": "on purpose", "kind": "code",
                            "spec": "Finish the delivery.", "depends_on": [a, b]}]) == []
    assert dependency_ids(p.db.one("SELECT * FROM tasks WHERE title='on purpose'")) == [a, b]
