# A check, schedule or heal check whose script is rewritten in place while it runs (scriptwatch), and
# tt-project's own script writes, which rename a new file into place instead (project.durable_write).


def _self_rewriting(path, then_exit=0):
    """A bash script whose first run rewrites itself in place (same inode), as an editor saving over it
    does: the running shell reads on into the new text and fails with a syntax error. The new version
    exits `then_exit`. Each run appends a line to <path>.runs."""
    code = ('import sys; f = open(sys.argv[1], "r+"); '
            f'f.write("#!/bin/bash\\necho x >> \\"$0.runs\\"\\nexit {then_exit}\\n#" + ")" * 3000 + "\\n")')
    path.write_text(
        "#!/bin/bash\n"
        "echo x >> \"$0.runs\"\n"
        "if [ ! -e \"$0.ran\" ]; then\n"
        "  : > \"$0.ran\"\n"
        f"  python3 -c {shlex.quote(code)} \"$0\"\n"
        "fi\n"
        "echo first version done\n")
    path.chmod(0o755)
    return path


def _runs(path):
    runs = path.with_name(path.name + ".runs")
    return len(runs.read_text().splitlines()) if runs.exists() else 0


def test_scriptwatch_reruns_once_a_command_whose_script_was_rewritten_in_place_mid_run(tmp_path):
    from ttp import scriptwatch
    script = _self_rewriting(tmp_path / "check.sh")
    ino = script.stat().st_ino
    run = lambda: subprocess.run(["bash", "-c", "bash check.sh"], cwd=tmp_path, capture_output=True, text=True)
    said = []
    out = scriptwatch.rerun(run, lambda r: r.returncode != 0, "bash check.sh", tmp_path, None, said.append)
    assert script.stat().st_ino == ino, "the script was edited in place, as in the reported failure"
    assert out.returncode == 0 and _runs(script) == 2
    assert said == [f"{scriptwatch.LABEL}: check.sh"]
    # Without the watch the same first run is the plain failure the report showed.
    (tmp_path / "check.sh.ran").unlink()
    (tmp_path / "check.sh.runs").unlink()
    _self_rewriting(script)
    first = run()
    assert first.returncode == 2 and "syntax error near unexpected token" in first.stderr


def test_scriptwatch_still_fails_a_real_failure_and_a_second_failure(tmp_path):
    from ttp import scriptwatch
    fail = tmp_path / "fail.sh"
    fail.write_text("#!/bin/bash\necho x >> \"$0.runs\"\nexit 1\n")
    said = []
    run = lambda name: lambda: subprocess.run(["bash", name], cwd=tmp_path).returncode
    # Nothing changed: the failure is reported as before, run once.
    assert scriptwatch.rerun(run("fail.sh"), bool, "bash fail.sh", tmp_path, None, said.append) == 1
    assert _runs(fail) == 1 and said == []
    # Changed mid-run, and the new version fails too: the rerun's failure counts.
    twice = _self_rewriting(tmp_path / "twice.sh", then_exit=3)
    assert scriptwatch.rerun(run("twice.sh"), bool, "bash twice.sh", tmp_path, None, said.append) == 3
    assert _runs(twice) == 2 and said == [f"{scriptwatch.LABEL}: twice.sh"]
    # A command that names no script, or no command (a remote check), is never rerun.
    count = tmp_path / "count"
    sh = lambda: subprocess.run(["sh", "-c", f"echo x >> {count}; exit 1"]).returncode
    assert scriptwatch.rerun(sh, bool, f"echo x >> {count}; exit 1", tmp_path) == 1
    assert scriptwatch.rerun(sh, bool, None, tmp_path) == 1
    assert len(count.read_text().splitlines()) == 2


def test_scriptwatch_watches_named_scripts_their_neighbours_and_scripts_on_path(tmp_path):
    from ttp import scriptwatch
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, text in (("check", "#!/bin/sh\n"), ("helper", "#!/bin/sh\n"), ("lib.sh", "x=1\n"), ("data.txt", "x\n")):
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    (tmp_path / "test_a.py").write_text("def test_a(): pass\n")
    got = {p.name for p in scriptwatch.scripts("bin/check && python3 -m pytest test_a.py", tmp_path, {"PATH": ""})}
    assert got == {"check", "helper", "lib.sh"}   # a data file or a test module is no script
    assert {p.name for p in scriptwatch.scripts("check --fast", tmp_path, {"PATH": str(bin_)})} == {"check", "helper", "lib.sh"}
    seen = scriptwatch.watch("bin/check", tmp_path, {"PATH": ""})
    assert scriptwatch.changed(seen) == []
    tmp = bin_ / ".helper.tmp"
    tmp.write_text("#!/bin/sh\nexit 0\n")
    os.replace(tmp, bin_ / "helper")   # replaced by rename counts as well
    assert scriptwatch.changed(seen) == [str(bin_ / "helper")]


def test_ttp_checks_reruns_a_check_whose_script_was_rewritten_mid_run_and_labels_it(env, tmp_path, monkeypatch, capsys):
    from ttp import cli, scriptwatch
    p, repo, run, git = _checks_repo(env, tmp_path, monkeypatch)
    _self_rewriting(repo / "check.sh")
    (repo / ".gitignore").write_text("*.ran\n*.runs\n")
    subprocess.run([*git, "add", "check.sh", ".gitignore"], check=True)
    subprocess.run([*git, "commit", "-qm", "check"], check=True)
    p.set_config("delivery.push_checks", [{"run": "bash check.sh", "cwd": "authoring"}])
    cli.main(["checks"])
    out = capsys.readouterr().out
    assert f"ttp checks: {scriptwatch.LABEL}: check.sh" in out
    assert json.loads((run / "checks.json").read_text())["passed"] and _runs(repo / "check.sh") == 2
    assert scriptwatch.LABEL in (run / "checks.log").read_text()
    # A check that fails with its script unchanged still fails, run once.
    (repo / "fail.sh").write_text("#!/bin/bash\necho x >> \"$0.runs\"\nexit 1\n")
    subprocess.run([*git, "add", "-A"], check=True)   # with check.sh as its first run rewrote it
    subprocess.run([*git, "commit", "-qm", "fail"], check=True)
    p.set_config("delivery.push_checks", [{"run": "bash fail.sh", "cwd": "authoring"}])
    with pytest.raises(SystemExit):
        cli.main(["checks"])
    got = capsys.readouterr()
    assert scriptwatch.LABEL not in got.out and _runs(repo / "fail.sh") == 1, got
    assert not json.loads((run / "checks.json").read_text())["passed"]


def test_a_push_check_whose_script_was_rewritten_mid_run_is_rerun_once(tmp_path):
    from ttp import push, scriptwatch
    _self_rewriting(tmp_path / "check.sh")
    said = []
    assert push._run_one("bash check.sh", tmp_path, dict(os.environ), said.append) == 0
    assert said == [f"{scriptwatch.LABEL}: check.sh"]
    _self_rewriting(tmp_path / "again.sh", then_exit=1)
    assert push._run_one("bash again.sh", tmp_path, dict(os.environ), said.append) == push.CHECKS_FAILED
    assert _runs(tmp_path / "again.sh") == 2


def test_a_schedule_command_and_a_heal_check_whose_script_changed_mid_run_are_rerun_once(env, tmp_path):
    from ttp import heal, schedrun, scriptwatch
    _self_rewriting(tmp_path / "watch.sh")
    out = schedrun.run_group("bash watch.sh", cwd=str(tmp_path), timeout=60, env=dict(os.environ))
    assert out.returncode == 0 and out.stderr.startswith(f"ttp: {scriptwatch.LABEL}: watch.sh\n")
    assert _runs(tmp_path / "watch.sh") == 2
    _self_rewriting(tmp_path / "healthy.sh")
    rc, text = heal.check({"check": "bash healthy.sh", "timeout_s": 60}, {}, str(tmp_path), dict(os.environ))
    assert rc == 0 and text.endswith(f"{scriptwatch.LABEL}: healthy.sh")
    # An unhealthy answer with nothing changed is the check's answer, run once.
    (tmp_path / "sick.sh").write_text("#!/bin/bash\necho x >> \"$0.runs\"\necho down\nexit 1\n")
    assert heal.check({"check": "bash sick.sh", "timeout_s": 60}, {}, str(tmp_path), dict(os.environ)) == (1, "down")
    assert _runs(tmp_path / "sick.sh") == 1


def test_a_watcher_or_precheck_exit_1_is_its_answer_and_never_rerun_when_a_neighbour_changed(env):
    from ttp import daemon as dm, schedrun, scriptwatch
    p = make(env)
    bin_dir = p.root / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "helper.sh").write_text("#!/bin/bash\ntrue\n")
    # A watcher's first run reports, replaces a neighbour script by an atomic rename and exits 1 ("found
    # something"); a rerun would print nothing and its report would be lost.
    (bin_dir / "watch").write_text(
        "#!/bin/bash\n"
        "echo x >> \"$0.runs\"\n"
        "if [ ! -e \"$0.ran\" ]; then\n"
        "  : > \"$0.ran\"\n"
        "  printf '#!/bin/bash\\nfalse\\n' > bin/helper.sh.tmp && mv bin/helper.sh.tmp bin/helper.sh\n"
        "  echo 'chip dropped'\n"
        "  exit 1\n"
        "fi\n")
    d = dm.Daemon(p.base)
    assert d._run_command_watcher({"name": "chips"}, {"command": "bash bin/watch"}) == "ok (1 observations)"
    assert _runs(bin_dir / "watch") == 1
    # The schedule precheck's exit 1 ("the run goes ahead") is an answer too: run once.
    (bin_dir / "watch.ran").unlink()
    (bin_dir / "watch.runs").unlink()
    out = schedrun.run_group("bash bin/watch", cwd=str(p.root), timeout=60, env=dict(os.environ), ok=(0, 1))
    assert out.returncode == 1 and out.stdout == "chip dropped\n" and _runs(bin_dir / "watch") == 1
    # A real failure (a bash syntax error from a script rewritten in place) is still rerun once.
    _self_rewriting(bin_dir / "broken.sh")
    out = schedrun.run_group("bash bin/broken.sh", cwd=str(p.root), timeout=60, env=dict(os.environ), ok=(0, 1))
    assert out.returncode == 0 and out.stderr.startswith(f"ttp: {scriptwatch.LABEL}: bin/broken.sh\n")
    assert _runs(bin_dir / "broken.sh") == 2


def test_script_writes_rename_a_new_file_into_place_keep_the_mode_and_leave_no_partial_file(env, tmp_path, monkeypatch):
    from ttp import cli, project
    f = tmp_path / "bin" / "check"
    f.parent.mkdir()
    f.write_text("#!/bin/sh\necho old\n")
    f.chmod(0o750)
    reader = open(f)   # a shell running the old version keeps reading the old file whole
    ino = f.stat().st_ino
    project.durable_write(f, "#!/bin/sh\necho new\n")
    assert f.stat().st_ino != ino and stat.S_IMODE(f.stat().st_mode) == 0o750
    assert reader.read() == "#!/bin/sh\necho old\n" and f.read_text() == "#!/bin/sh\necho new\n"
    reader.close()
    # A write that fails part way leaves the old file whole and no temporary file.
    real = os.replace
    monkeypatch.setattr(project.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        project.durable_write(f, "#!/bin/sh\necho half\n")
    monkeypatch.setattr(project.os, "replace", real)
    assert f.read_text() == "#!/bin/sh\necho new\n" and [x.name for x in f.parent.iterdir()] == ["check"]
    # Installing a template tree over a live one (ttp setup, init, upgrade) renames each file in too.
    src = tmp_path / "src"
    src.mkdir()
    (src / "check").write_text("#!/bin/sh\necho installed\n")
    (src / "check").chmod(0o755)
    ino = f.stat().st_ino
    cli._copy_tree(src, f.parent)
    assert f.stat().st_ino != ino and f.read_text() == "#!/bin/sh\necho installed\n"
    assert stat.S_IMODE(f.stat().st_mode) == 0o755 and [x.name for x in f.parent.iterdir()] == ["check"]
