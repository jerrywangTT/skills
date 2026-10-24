"""Packaging and script behaviour for the tt-model-op-analysis plugin."""

from __future__ import annotations

import csv
import json
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = REPO / "plugins" / "tt-model-op-analysis"
SCRIPTS = PLUGIN / "scripts"
SKILL_NAMES = ("model-op-analysis", "upload-op-analysis")
PHASES = ("static-phase.md", "measured-source.md", "measured-destination.md")


def test_plugin_skills_and_phase_references():
    assert sorted(p.name for p in (PLUGIN / "skills").iterdir()) == list(SKILL_NAMES)
    for name in PHASES:
        assert (PLUGIN / "references" / name).is_file()


def test_manifest_versions_match():
    claude = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    codex = json.loads((PLUGIN / ".codex-plugin" / "plugin.json").read_text())
    assert claude["version"] == codex["version"] == "0.11.0"


def test_shared_setup_is_step_by_step():
    setup = (PLUGIN / "references" / "run-setup.md").read_text()
    assert "Ask one decision per turn and stop for its answer" in setup
    assert "Ask which model to analyse. Do no model-specific discovery before the answer" in setup


def test_setup_asks_source_first_then_model_ref_implementation_test():
    setup = (PLUGIN / "references" / "run-setup.md").read_text()
    order = setup[setup.index("### Setup order"):setup.index("Destinations are asked after static analysis")]
    steps = ["probe the local device", "Ask for the source architecture", "Ask which model to analyse",
             "ask which ref to pin", "Ask which implementation to analyse", "Rank the whole-model"]
    assert [order.index(s) for s in steps] == sorted(order.index(s) for s in steps)
    assert "recommend the\n   probed local device's profile" in order
    assert "Never hide a\n   directory" in order
    flow = (PLUGIN / "skills" / "model-op-analysis" / "SKILL.md").read_text()
    heads = ["**Preconditions, first.**", "**Source architecture.**", "**Model and inventory.**", "**New run.**"]
    assert [flow.index(h) for h in heads] == sorted(flow.index(h) for h in heads)


def test_device_runs_are_sequential():
    skill = (PLUGIN / "skills" / "model-op-analysis" / "SKILL.md").read_text()
    assert "Device runs are sequential: one device job at a time" in skill
    assert "never give device runs to parallel subagents" in skill
    source = (PLUGIN / "references" / "measured-source.md").read_text()
    assert "one after the other" in source and "never two at once" in source


def test_destination_is_a_projection_of_source_rows():
    static = (PLUGIN / "references" / "static-phase.md").read_text()
    section = static[static.index("## Adding a destination"):]
    assert "always a projection of the source rows" in section
    assert "never offer a separate run instead of a\ndestination" in section
    status = (PLUGIN / "references" / "target-status.md").read_text()
    assert "neither drops the row nor makes it `n/a`" in status
    measured = (PLUGIN / "references" / "measured-destination.md").read_text()
    assert "each with\nthe parameters written in the test" in measured
    assert "no script compares them" in measured


def test_setup_traps_cover_fit_by_code_not_name():
    setup = (PLUGIN / "references" / "run-setup.md").read_text()
    traps = setup[setup.index("### Setup traps"):setup.index("### No whole-model test")]
    assert "A directory's name is not its fit" in traps and "cite the one that decides" in traps
    assert "A defined skip helper is not a skip" in traps
    assert "never from earlier runs" in traps and "tests/pipeline_reorg" in traps


def test_earlier_runs_offer_migration_before_a_new_run():
    setup = (PLUGIN / "references" / "run-setup.md").read_text()
    section = setup[setup.index("## 4. Migrating a run"):]
    assert section.index("**Migrate** (recommended)") < section.index("**New run**")
    assert "no new trace of the model and no new device run" in section
    assert "do not rerun the device" in section
    for skill in SKILL_NAMES:
        assert '"Migrating a run"' in (PLUGIN / "skills" / skill / "SKILL.md").read_text()


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          capture_output=True, text=True)


def run_merge(*args: str) -> subprocess.CompletedProcess[str]:
    return run_script("merge_measured.py", *args, "--tracy-mode", "live")


def write_rows(path: pathlib.Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def op_row(i: int, launches_p150: int, port: str = "⚠️", port_ev: str = "fork x, no variant test",
           **overrides: str) -> dict:
    row = {
        "id": str(i), "stage": f"encoder {i}", "origin": "static", "ttnn_api": f"ttnn.op{i}",
        "op_code": f"Op{i}DeviceOperation",
        "device_op": f"prim::op{i}", "program_factory": f"Op{i}Factory", "call_site": f"model.py:{10 + i}",
        "launch_condition": "always", "confidence": "verified", "dispatch_evidence": f"op{i}.cpp select",
        "grid_dependency": "automatic",
        "p150:shapes": "(1,32)", "p150:launches": str(launches_p150), "p150:status": "✅",
        "p150:evidence": "entry test passes in CI",
        "quasar-bf16:tests": "",
        "quasar-bf16:test_result": "not run", "quasar-bf16:status": port, "quasar-bf16:evidence": port_ev,
    }
    row.update(overrides)
    return row


BLOCKER_FIELDS = ["id", "blocker", "affected_ids", "evidence", "required_change"]
HOST_FIELDS = ["id", "host_work", "when", "counts_as_host_op", "reason"]


def make_static_run(tmp_path: pathlib.Path, op_rows: list[dict], trace_rows: list[dict],
                    destination: str | None = "quasar-bf16", blockers: list[dict] | None = None,
                    host_ops: list[dict] | None = None) -> pathlib.Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"kind": "static", "source": "p150",
                                              "destinations": [destination] if destination else [],
                                              "stage_blocks": ["encoder", "s1", "s2"],
                                              "configurable_grid": {"value": "yes", "evidence": "m.py:1 grid from device"},
                                              "configurable_layers": {"value": "n/a", "evidence": "no layer stack"},
                                              "entry_test_issues": {"checked": "2026-10-01", "open": []},
                                              "analysed_by": [{"agent": "Claude Code", "model": "m",
                                                               "work": "static trace", "date": "2026-10-01"}],
                                              "entry_test_ci": {"value": "no", "evidence": "no workflow runs it on P150"}}))
    write_rows(run / "op_table.csv", op_rows)
    write_rows(run / "call_trace.csv", trace_rows)
    write_rows(run / "port_blockers.csv", blockers or [], BLOCKER_FIELDS)
    write_rows(run / "host_ops.csv", host_ops or [], HOST_FIELDS)
    return run


def trace_row(i: int, repeats: int, per_repeat: int, profile: str = "p150") -> dict:
    return {"id": str(i), "profile": profile, "stage": "encoder", "ops": "op1, op2",
            "repeats": str(repeats), "launches_per_repeat": str(per_repeat), "notes": ""}


def validate(run: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return run_script("validate_static.py", str(run))


def test_valid_static_run_passes_and_prints_range(tmp_path):
    rows = [op_row(1, 12), op_row(2, 1, launch_condition="skipped when the buffer does not move")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 12, 1), trace_row(2, 1, 1)])
    write_support(run, [("prim::op1", "⚠️", "fork x, no test"), ("prim::op2", "⚠️", "fork y, no test")])
    result = validate(run)
    assert result.returncode == 0, result.stdout
    assert result.stdout.splitlines() == ["OK", "p150: 13 launches per inference, 1 of them conditional (12-13); 0 measured-only launches"]


SUPPORT_FIELDS = ["device_op", "quasar-bf16:op_support", "quasar-bf16:op_evidence"]


def write_support(run: pathlib.Path, verdicts: list[tuple[str, str, str]]) -> None:
    write_rows(run / "device_op_support.csv", [dict(zip(SUPPORT_FIELDS, v)) for v in verdicts], SUPPORT_FIELDS)


def support_from_rows(run: pathlib.Path) -> None:
    """A verdict per device op that agrees with its row statuses, for tests about other checks."""
    statuses: dict[str, set[str]] = {}
    for r in read(run / "op_table.csv"):
        statuses.setdefault(r["device_op"] or r["op_code"], set()).add(r["quasar-bf16:status"])
    verdicts = []
    for op, s in statuses.items():
        verdict = ("✅" if "✅" in s else "n/a" if s == {"n/a"} else "❌" if s <= {"❌", "n/a"} else "⚠️")
        verdicts.append((op, verdict, "test"))
    write_support(run, verdicts)


def test_device_op_support_missing_or_undecided_warns(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1), op_row(2, 1)], [trace_row(1, 1, 2)])
    result = validate(run)
    assert result.returncode == 0 and "WARNING: device_op_support.csv missing" in result.stdout
    write_support(run, [("prim::op1", "", "")])
    out = validate(run).stdout
    assert "no row for prim::op2" in out and "no verdict for prim::op1 on quasar-bf16" in out and "ERROR" not in out


def test_device_op_support_must_agree_with_row_statuses(tmp_path):
    rows = [op_row(1, 1, port="✅", port_ev="test a"), op_row(2, 1, port="⚠️"), op_row(3, 1, port="❌", port_ev="xfail"),
            op_row(4, 1, port="❌", port_ev="xfail")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 4)])
    write_support(run, [("prim::op1", "❌", "x"), ("prim::op2", "❌", "x"), ("prim::op3", "n/a", "x"),
                        ("prim::op4", "❌", ""), ("prim::nope", "✅", "x")])
    out = validate(run).stdout
    assert "row 1: quasar-bf16:op_support '❌' but a row of the op is ✅" in out
    assert "row 2: quasar-bf16:op_support '❌' but rows of the op are ['⚠️']" in out
    assert "row 3: quasar-bf16:op_support 'n/a' with row statuses ['❌']" in out
    assert "row 4: quasar-bf16:op_support without quasar-bf16:op_evidence" in out
    assert "row 5: device_op 'prim::nope' is in no op_table row" in out
    # A verdict for the op as a whole may be ✅ where none of the model's calls is.
    write_support(run, [("prim::op1", "✅", "test a"), ("prim::op2", "⚠️", "fork, no test"),
                        ("prim::op3", "✅", "t.py::other_shape passes"), ("prim::op4", "❌", "no factory: f.cpp:3")])
    assert validate(run).returncode == 0


def test_device_op_support_short_row_is_reported_not_crashing(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    (run / "device_op_support.csv").write_text(",".join(SUPPORT_FIELDS) + "\nprim::op1,⚠️\n", encoding="utf-8")
    result = validate(run)
    assert "device_op_support.csv row 1: field count differs from the header" in result.stdout, result.stderr


def test_legacy_targets_list_is_rejected(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    (run / "run.json").write_text(json.dumps({"targets": ["p150", "quasar"]}))
    assert "predates source/destination targets" in validate(run).stdout


def test_source_must_be_a_profile_and_differ(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    (run / "run.json").write_text(json.dumps({"source": "quasar-bf16", "destinations": ["quasar-bf16"],
                                              "stage_blocks": ["encoder"]}))
    out = validate(run).stdout
    assert "source 'quasar-bf16' not in" in out
    assert "destination 'quasar-bf16' equals the source" in out


def test_origin_values_checked(tmp_path):
    out = validate(make_static_run(tmp_path, [op_row(1, 1, origin="guess")], [trace_row(1, 1, 1)])).stdout
    assert "row 1: origin 'guess' not in" in out


def test_measured_rows_excluded_from_trace_total(tmp_path):
    rows = [op_row(1, 2), op_row(2, 3, origin="measured", call_site="not traced", grid_dependency="not traced")]
    result = validate(make_static_run(tmp_path, rows, [trace_row(1, 1, 2)]))
    assert result.returncode == 0, result.stdout
    assert "p150: 2 launches per inference" in result.stdout
    assert "3 measured-only launches" in result.stdout


def test_test_result_must_match_status(tmp_path):
    rows = [op_row(1, 1, **{"quasar-bf16:tests": "t.py::a", "quasar-bf16:test_result": "fail"}),
            op_row(2, 1, port="✅", **{"quasar-bf16:tests": "t.py::b", "quasar-bf16:test_result": "pass"})]
    out = validate(make_static_run(tmp_path, rows, [trace_row(1, 2, 1)])).stdout
    assert "row 1: quasar-bf16:test_result 'fail' requires status ❌" in out
    assert "row 2:" not in out


def test_tests_column_needs_node_ids(tmp_path):
    rows = [op_row(1, 1, **{"quasar-bf16:tests": "some test"})]
    out = validate(make_static_run(tmp_path, rows, [trace_row(1, 1, 1)])).stdout
    assert "row 1: quasar-bf16:tests 'some test' is not a list of pytest node ids" in out


def test_pre_origin_run_is_rejected(tmp_path):
    rows = [op_row(1, 1)]
    del rows[0]["origin"]
    out = validate(make_static_run(tmp_path, rows, [trace_row(1, 1, 1)])).stdout
    assert "op_table.csv missing column origin" in out
    assert "upgrade_run.py" not in out


def test_off_by_one_id_fails(tmp_path):
    rows = [op_row(1, 12), op_row(2, 1)]
    rows[1]["id"] = "3"
    run = make_static_run(tmp_path, rows, [trace_row(1, 12, 1), trace_row(2, 1, 1)])
    result = validate(run)
    assert result.returncode == 1
    assert "op_table.csv row 2 has id 3" in result.stdout


def test_launch_total_mismatch_fails(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 12), op_row(2, 1)], [trace_row(1, 12, 1), trace_row(2, 2, 1)])
    result = validate(run)
    assert result.returncode == 1
    assert "p150: op_table launches 13 != call_trace launches 14" in result.stdout


def test_missing_and_unexpected_columns_fail(tmp_path):
    rows = [op_row(1, 1)]
    rows[0]["evidence"] = rows[0].pop("dispatch_evidence")
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 1)])
    out = validate(run).stdout
    assert "op_table.csv missing column dispatch_evidence" in out
    assert "op_table.csv unexpected column evidence" in out


def test_side_tables_follow_schema(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    write_rows(run / "host_ops.csv", [{"id": "1", "stage": "s", "operation": "o"}])
    assert "host_ops.csv missing column host_work" in validate(run).stdout


def test_short_row_is_reported_not_crashing(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1, port_ev="fork")], [trace_row(1, 1, 1)])
    text = (run / "op_table.csv").read_text(encoding="utf-8").splitlines()
    (run / "op_table.csv").write_text("\n".join([text[0], text[1].rsplit(",", 1)[0]]) + "\n", encoding="utf-8")
    result = validate(run)
    assert result.returncode == 1
    assert "op_table.csv row 1: field count differs from the header" in result.stdout
    assert "Traceback" not in result.stderr


def test_every_status_needs_its_own_evidence(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1, port_ev="")], [trace_row(1, 1, 1)])
    assert "op_table.csv row 1: quasar-bf16:status without quasar-bf16:evidence" in validate(run).stdout


def test_port_only_status_value_is_gone(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1, port="variant not used by port")], [trace_row(1, 1, 1)])
    assert "quasar-bf16:status 'variant not used by port' not in" in validate(run).stdout


def test_unknown_status_fails(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1, **{"p150:status": "ok"})], [trace_row(1, 1, 1)])
    assert "op_table.csv row 1: p150:status 'ok' not in" in validate(run).stdout


def test_na_exactly_when_no_launches(tmp_path):
    rows = [op_row(1, 1, **{"p150:status": "n/a"}), op_row(2, 0)]
    out = validate(make_static_run(tmp_path, rows, [trace_row(1, 1, 1)])).stdout
    assert "row 1: p150 has 1 launches with status 'n/a'" in out
    assert "row 2: p150 has 0 launches with status '✅'" in out


def test_bad_launch_count_reported_once(tmp_path):
    out = validate(make_static_run(tmp_path, [op_row(1, 1, **{"p150:launches": "2x"})],
                                   [trace_row(1, 1, 1)])).stdout
    assert out.count("'2x' is not an integer") == 1
    assert "launches with status" not in out


def test_call_site_needs_line(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1, call_site="model.py: use_transpose_as_fold=True")],
                          [trace_row(1, 1, 1)])
    assert "call_site 'model.py: use_transpose_as_fold=True' has no <file>:<line>" in validate(run).stdout


def test_grid_dependency_values(tmp_path):
    rows = [op_row(1, 1, grid_dependency="profile-derived"), op_row(2, 1, grid_dependency="see bf16 note"),
            op_row(3, 1, grid_dependency="none"), op_row(4, 1, grid_dependency="8x4 CoreGrid, model.py:29"),
            op_row(5, 1, grid_dependency="automatic: 100 cores via determine_parallel_config conv2d_utils.cpp:201")]
    out = validate(make_static_run(tmp_path, rows, [trace_row(1, 5, 1)])).stdout
    assert "row 1: grid_dependency 'profile-derived'" in out
    assert "row 2: grid_dependency 'see bf16 note'" in out
    assert "row 3: grid_dependency" not in out and "row 4: grid_dependency" not in out
    assert "row 5: grid_dependency 'automatic: 100 cores" in out and "must be exactly 'automatic'" in out


def test_launch_condition_must_be_exact(tmp_path):
    rows = [op_row(1, 1, launch_condition="Always"), op_row(2, 1, launch_condition="")]
    out = validate(make_static_run(tmp_path, rows, [trace_row(1, 2, 1)])).stdout
    assert "row 1: launch_condition 'Always' must be exactly 'always'" in out
    assert "row 2: launch_condition empty" in out


def test_blocker_ids_must_exist(tmp_path):
    blockers = [{"id": "1", "blocker": "b", "affected_ids": "1, 9,", "evidence": "e", "required_change": "c"},
                {"id": "2", "blocker": "b", "affected_ids": "all", "evidence": "e", "required_change": "c"},
                {"id": "3", "blocker": "b", "affected_ids": "", "evidence": "e", "required_change": "c"}]
    out = validate(make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)], blockers=blockers)).stdout
    assert "port_blockers.csv row 1: affected id '9' is not an op_table id" in out
    assert "affected id ''" not in out
    assert "row 2:" not in out
    assert "port_blockers.csv row 3: affected_ids empty" in out


def test_host_op_flag_values(tmp_path):
    host = [{"id": "1", "host_work": "kwargs", "when": "per block", "counts_as_host_op": "maybe", "reason": "r"}]
    out = validate(make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)], host_ops=host)).stdout
    assert "counts_as_host_op 'maybe' not in" in out


def test_shared_evidence_passes_and_fake_or_long_evidence_warns(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    rows = [op_row(i, 1, port_ev="same text") for i in range(1, 6)]
    run = make_static_run(tmp_path / "a", rows, [trace_row(1, 5, 1)])
    support_from_rows(run)
    result = validate(run)
    assert result.returncode == 0 and "WARNING" not in result.stdout, result.stdout
    rows = [op_row(1, 1, port_ev="Row 1 (encoder 1): fork x"), op_row(2, 1, port_ev="x" * 301),
            op_row(3, 1, port_ev="TTsim run: t.py::a hang || " + "y" * 290)]
    result = validate(make_static_run(tmp_path / "b", rows, [trace_row(1, 3, 1)]))
    assert result.returncode == 0, result.stdout
    assert "quasar-bf16:evidence: rows 1 start with 'Row <n>'" in result.stdout
    assert "quasar-bf16:evidence: rows 2 exceed 300 characters" in result.stdout


def test_blackhole_destination_needs_its_trace(tmp_path):
    rows = [op_row(1, 2, **{"p100:shapes": "(1,32)", "p100:launches": "1", "p100:status": "✅",
                            "p100:evidence": "CI"})]
    for col in ("quasar-bf16:tests", "quasar-bf16:test_result",
                "quasar-bf16:status", "quasar-bf16:evidence"):
        del rows[0][col]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 2)], destination="p100")
    assert "p100: op_table launches 1 != call_trace launches 0" in validate(run).stdout
    write_rows(run / "call_trace.csv", [trace_row(1, 1, 2), trace_row(2, 1, 1, profile="p100")])
    assert validate(run).returncode == 0


def sheet(run: pathlib.Path, *args: str) -> dict:
    result = run_script("sheet_spec.py", str(run), *args)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def tab(spec: dict, title: str) -> list[list]:
    return next(t["values"] for t in spec["tabs"] if t["title"] == title)


def test_sheet_spec_layout(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1), op_row(2, 2)], [trace_row(1, 1, 3)])
    write_rows(run / "changes_op_table.csv", [{"change": "added", "key": "k", "column": "", "old": "", "new": ""}])
    spec = sheet(run, "--info-sheet-id", "7", "--first-sheet-id", "20")
    assert [(t["title"], t["sheetId"]) for t in spec["tabs"]] == [
        ("Info", 7), ("ttnn calls", 20), ("Launch order", 21), ("Port blockers", 22), ("Host work in code", 23),
        ("Changes: ttnn calls", 24)]
    reqs = spec["requests"]
    info = reqs[0]["updateSheetProperties"]["properties"]
    assert (info["sheetId"], info["title"]) == (7, "Info")
    assert info["gridProperties"]["columnCount"] >= 3
    added = [(r["addSheet"]["properties"]["title"], r["addSheet"]["properties"]["index"]) for r in reqs if "addSheet" in r]
    assert added == [("ttnn calls", 1), ("Launch order", 2), ("Port blockers", 3), ("Host work in code", 4),
                     ("Changes: ttnn calls", 5)]
    assert not any("autoResizeDimensions" in r for r in reqs)
    wrapped = {r["repeatCell"]["range"]["sheetId"] for r in reqs
               if "repeatCell" in r and r["repeatCell"]["cell"]["userEnteredFormat"].get("wrapStrategy") == "WRAP"}
    assert wrapped == {7, 20, 21, 22, 23, 24}
    assert tab(spec, "Port blockers")[0][3] == "Evidence"


def test_sheet_spec_op_table_is_a_compact_view(tmp_path):
    rows = [op_row(1, 1, call_site="models/demos/x/tt/model.py:11; ttnn/cpp/ttnn/operations/a/b.cpp:5",
                   port_ev="Row 1 (encoder 1): fork x at ttnn/cpp/ttnn/operations/q/c.cpp:9"),
            op_row(2, 2, port_ev="fork y")]
    spec = sheet(make_static_run(tmp_path, rows, [trace_row(1, 1, 3)]))
    ops = tab(spec, "ttnn calls")
    headers = ops[0]
    assert headers[:4] == ["#", "Stage", "ttnn API", "Device op / program factory"]
    # one value in every row: dropped from the table, stated once on Info
    for dropped in ("Origin (static / measured)", "P150: evidence and gap", "Launch condition"):
        assert dropped not in headers
    # status and launch columns stay even when constant
    assert "P150: status" in headers and "P150: launches per inference" in headers
    row = dict(zip(headers, ops[1]))
    assert row["Device op / program factory"] == "prim::op1 / Op1Factory"
    assert row["Call site (file:line)"] == "tt/model.py:11; a/b.cpp:5"
    assert row["Quasar, BF16 projection: evidence and gap"] == "fork x at q/c.cpp:9"
    info = tab(spec, "Info")
    assert ["P150: evidence and gap", "entry test passes in CI"] in info
    assert ["Origin (static / measured)", "static"] in info


def test_sheet_spec_counts_are_formulas_with_expected_values(tmp_path):
    rows = [op_row(1, 3, port="❌", port_ev="fails"),
            op_row(2, 1, launch_condition="skipped when the buffer does not move")]
    spec = sheet(make_static_run(tmp_path, rows, [trace_row(1, 1, 4)]))
    info = tab(spec, "Info")
    counts = next(i for i, r in enumerate(info) if r[:1] == ["Counts"])
    by_name = {r[0]: (i + 1, r) for i, r in enumerate(info) if r and i > counts}
    row_number, row = by_name["All rows"]
    assert row[1].startswith("=COUNTA(") and row[2].startswith("=SUM(")
    assert spec["expected"][f"Info!B{row_number}"] == 2 and spec["expected"][f"Info!C{row_number}"] == 4
    row_number, _ = by_name["Launch condition: always"]
    assert spec["expected"][f"Info!C{row_number}"] == 3
    row_number, _ = by_name["Quasar, BF16 projection: ❌"]
    assert spec["expected"][f"Info!B{row_number}"] == 1
    assert "Quasar, BF16 projection: n/a" not in by_name
    # The launch-order total equals the ttnn calls total by the validator; Info does not repeat it.
    assert "Launch order launches (P150)" not in by_name


def test_sheet_spec_info_reads_summary_sections(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    (run / "summary.md").write_text("# M kernel op analysis\n\n| a | b |\n|---|---|\n\n## Findings\n\n"
                                    "- Launches: **120** per inference\n- plain bullet\n\n1. first\n")
    info = tab(sheet(run), "Info")
    assert info[0] == ["M kernel op analysis"]
    assert ["Findings"] in info and ["Launches", "'120 per inference"] in info
    assert ["", "plain bullet"] in info and ["'1.", "first"] in info


TENSOR_FIELDS = ["W_PAD[LOGICAL]", "Z_PAD[LOGICAL]", "Y_PAD[LOGICAL]", "X_PAD[LOGICAL]",
                 "LAYOUT", "DATATYPE", "MEMORY"]
OPS_HEADER = (["OP CODE", "OP TYPE", "ATTRIBUTES", "CORE COUNT", "HOST START TS", "HOST END TS",
               "HOST DURATION [ns]", "DEVICE KERNEL DURATION [ns]"]
              + [f"INPUT_{i}_{f}" for i in range(2) for f in TENSOR_FIELDS]
              + [f"OUTPUT_0_{f}" for f in TENSOR_FIELDS])
TENSOR_A = ["1[1]", "10[10]", "32[4]", "64[64]", "TILE", "BFLOAT16", "DEV_1_L1_HEIGHT_SHARDED"]
NO_TENSOR = [""] * len(TENSOR_FIELDS)


def ops_csv(path: pathlib.Path, rows: list[list]) -> pathlib.Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(OPS_HEADER)
        writer.writerows(rows)
    return path


def dev(code, start, end, cores, kernel_ns, host_ns=100):
    return [code, "tt_dnn_device", "{'program_config': 'MatmulMultiCoreReuseMultiCastProgramConfig'}",
            cores, start, end, host_ns, kernel_ns] + TENSOR_A + NO_TENSOR + TENSOR_A


def host(code, start, end, host_ns):
    return [code, "python_fallback", "", "", start, end, host_ns, ""] + NO_TENSOR * 3


def sign(name, ts):
    return [name, "signpost", "", "", ts, "", "", ""] + NO_TENSOR * 3


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def test_warm_window_uses_second_iteration_only(tmp_path):
    src = ops_csv(tmp_path / "ops.csv", [
        dev("MatmulDeviceOperation", 0, 10, 120, 5_000_000),
        sign("start", 20),
        dev("MatmulDeviceOperation", 1_000_000, 2_000_000, 120, 1_000_000),
        host("TorchPermute", 2_000_000, 3_000_000, 1_000_000),
        dev("SoftmaxDeviceOperation", 3_000_000, 5_000_000, 64, 500_000),
        sign("end", 6_000_000),
    ])
    out = tmp_path / "out"
    result = run_script("tracy_report.py", str(src), str(out), "--start", "start", "--end", "end")
    assert result.returncode == 0, result.stdout + result.stderr
    measured = read(out / "measured_ops.csv")
    assert [r["op_code"] for r in measured] == ["MatmulDeviceOperation", "TorchPermute", "SoftmaxDeviceOperation"]
    fb = read(out / "host_fallback.csv")[0]
    assert fb["window"] == "warm (signposts start..end)"
    assert (fb["device_ops"], fb["host_ops"]) == ("2", "1")
    assert fb["device_op_time_ms"] == "1.500"
    assert fb["host_op_time_ms"] == "1.000"
    assert fb["total_time_ms"] == "4.000"
    assert fb["host_pct_of_total"] == "25.00"
    assert "threshold" not in (out / "host_fallback.csv").read_text()


def test_inputs_outputs_from_expanded_columns(tmp_path):
    src = ops_csv(tmp_path / "ops.csv", [dev("MatmulDeviceOperation", 0, 1, 8, 1)])
    out = tmp_path / "out"
    assert run_script("tracy_report.py", str(src), str(out)).returncode == 0
    row = read(out / "measured_ops.csv")[0]
    tensor = "1[1]x10[10]x32[4]x64[64] TILE BFLOAT16 DEV_1_L1_HEIGHT_SHARDED"
    assert row["inputs"] == tensor
    assert row["outputs"] == tensor


def test_no_signposts_is_cold_whole_csv(tmp_path):
    src = ops_csv(tmp_path / "ops.csv", [dev("MatmulDeviceOperation", 0, 1_000_000, 8, 1_000_000)])
    out = tmp_path / "out"
    result = run_script("tracy_report.py", str(src), str(out))
    assert result.returncode == 0, result.stderr
    assert read(out / "host_fallback.csv")[0]["window"] == "cold (no signposts)"
    assert len(read(out / "measured_ops.csv")) == 1


def test_missing_named_signpost_fails(tmp_path):
    src = ops_csv(tmp_path / "ops.csv", [dev("MatmulDeviceOperation", 0, 1, 8, 1)])
    result = run_script("tracy_report.py", str(src), str(tmp_path / "out"), "--start", "start", "--end", "end")
    assert result.returncode == 2
    assert "ERROR: signpost 'start' not found" in result.stdout


def test_footprint_max_cores_and_unmeasured_dram(tmp_path):
    src = ops_csv(tmp_path / "ops.csv", [
        dev("MatmulDeviceOperation", 0, 1, 64, 1), dev("MatmulDeviceOperation", 1, 2, 120, 1),
    ])
    out = tmp_path / "out"
    assert run_script("tracy_report.py", str(src), str(out)).returncode == 0
    fp = read(out / "footprint.csv")
    assert fp == [{"op_code": "MatmulDeviceOperation", "launches": "2", "max_core_count": "120",
                   "peak_dram_mb": "not measured"}]


def test_footprint_uses_given_peak_dram(tmp_path):
    src = ops_csv(tmp_path / "ops.csv", [dev("MatmulDeviceOperation", 0, 1, 64, 1)])
    out = tmp_path / "out"
    assert run_script("tracy_report.py", str(src), str(out), "--peak-dram-mb", "198").returncode == 0
    assert read(out / "footprint.csv")[0]["peak_dram_mb"] == "198.0"


MEASURED_FIELDS = ["id", "op_code", "op_type", "attributes", "inputs", "outputs",
                   "core_count", "device_kernel_ns", "host_ns"]


def static_table(path, rows):
    base = {"stage": "s", "ttnn_api": "a", "device_op": "d", "call_site": "", "dispatch_evidence": "e",
            "confidence": "verified", "grid_dependency": "", "p150:shapes": "", "p150:status": "✅"}
    write_rows(path, [{"id": str(i), **base, **r} for i, r in enumerate(rows, 1)])
    return path


def measured_table(path, rows):
    write_rows(path, [{"id": str(i), "op_code": c, "op_type": "tt_dnn_device", "attributes": a,
                       "inputs": "", "outputs": "", "core_count": "1", "device_kernel_ns": "1", "host_ns": "1"}
                      for i, (c, a) in enumerate(rows, 1)], fieldnames=MEASURED_FIELDS)
    return path


def run_diff(tmp_path, static_rows, measured_rows):
    s = static_table(tmp_path / "op_table.csv", static_rows)
    m = measured_table(tmp_path / "measured_ops.csv", measured_rows)
    out = tmp_path / "diff.csv"
    result = run_script("diff_reports.py", "--static", str(s), "--measured", str(m),
                        "--profile", "p150", "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    return read(out)


def test_diff_categories(tmp_path):
    rows = run_diff(tmp_path,
        [{"op_code": "MatmulDeviceOperation", "program_factory": "MatmulMultiCoreReuseMcast2DProgramFactory (cfg)", "p150:launches": "2"},
         {"op_code": "SoftmaxDeviceOperation", "program_factory": "SoftmaxShardedProgramFactory", "p150:launches": "1"},
         {"op_code": "ReshardDeviceOperation", "program_factory": "ReshardGenericFactory", "p150:launches": "1"}],
        [("MatmulDeviceOperation", "MatmulMultiCoreReuseMcast2DProgramFactory"),
         ("SoftmaxDeviceOperation", "other"), ("SoftmaxDeviceOperation", "other"),
         ("TilizeDeviceOperation", "")])
    got = {(r["category"], r["op_code"]) for r in rows}
    assert got == {
        ("count differs", "MatmulDeviceOperation"),
        ("count differs", "SoftmaxDeviceOperation"),
        ("missing in measured", "ReshardDeviceOperation"),
        ("extra in measured", "TilizeDeviceOperation"),
    }


def test_static_row_without_op_code_reported(tmp_path):
    rows = run_diff(tmp_path, [{"op_code": "", "program_factory": "X", "p150:launches": "1"}], [])
    assert [(r["category"], r["static_ids"]) for r in rows] == [("unmatched static row", "1")]


def test_identical_tables_have_no_diff(tmp_path):
    rows = run_diff(tmp_path,
        [{"op_code": "MatmulDeviceOperation", "program_factory": "Mcast2D", "p150:launches": "1"}],
        [("MatmulDeviceOperation", "uses Mcast2D")])
    assert rows == []


def cmp_rows(**overrides):
    row = {"id": "1", "stage": "enc", "ttnn_api": "ttnn.linear", "device_op": "prim::matmul",
           "program_factory": "Mcast2D", "p150:launches": "12", "p150:status": "✅"}
    row.update(overrides)
    return row


def run_compare(tmp_path, old, new):
    write_rows(tmp_path / "old.csv", old)
    write_rows(tmp_path / "new.csv", new)
    out = tmp_path / "changes.csv"
    result = run_script("compare_runs.py", "--old", str(tmp_path / "old.csv"),
                        "--new", str(tmp_path / "new.csv"), "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    return read(out)


def test_reordered_rows_no_change(tmp_path):
    a = cmp_rows(id="1")
    b = cmp_rows(id="2", ttnn_api="ttnn.add", device_op="prim::binary_ng", program_factory="BinaryNg")
    assert run_compare(tmp_path, [a, b], [dict(b, id="1"), dict(a, id="2")]) == []


def test_duplicate_keys_reordered_no_change(tmp_path):
    a = cmp_rows(id="1", **{"p150:shapes": "64"})
    b = cmp_rows(id="2", **{"p150:shapes": "128"})
    assert run_compare(tmp_path, [a, b], [dict(b, id="1"), dict(a, id="2")]) == []


def test_missing_key_column_fails(tmp_path):
    write_rows(tmp_path / "old.csv", [{"id": "1", "profile": "p150", "stage": "s"}])
    write_rows(tmp_path / "new.csv", [{"id": "1", "profile": "p150", "stage": "s"}])
    result = run_script("compare_runs.py", "--old", str(tmp_path / "old.csv"), "--new", str(tmp_path / "new.csv"),
                        "--out", str(tmp_path / "changes.csv"))
    assert result.returncode == 2
    assert "ERROR: key column ttnn_api missing" in result.stdout


def test_added_removed_changed(tmp_path):
    old = [cmp_rows(), cmp_rows(ttnn_api="ttnn.add", device_op="prim::binary_ng", program_factory="BinaryNg")]
    new = [cmp_rows(**{"p150:launches": "13"}),
           cmp_rows(ttnn_api="ttnn.softmax", device_op="prim::softmax", program_factory="Sharded")]
    rows = run_compare(tmp_path, old, new)
    got = {(r["change"], r["column"], r["old"], r["new"]) for r in rows}
    assert ("changed", "p150:launches", "12", "13") in got
    assert any(r["change"] == "added" and "ttnn.softmax" in r["key"] for r in rows)
    assert any(r["change"] == "removed" and "ttnn.add" in r["key"] for r in rows)


REFERENCES = PLUGIN / "references"
LINE_CITATION = re.compile(r"\.(?:cpp|hpp|h|py|rst|sh)\s*:\s*\d+")
PLUGIN_ROOT_REF = re.compile(r"<plugin-root>/((?:references|scripts)/[\w./-]+)")


def _unfenced_lines(path):
    fenced = False
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if not fenced and not line.lstrip().startswith(">"):
            yield n, line


def test_references_cite_no_line_numbers():
    paths = sorted(REFERENCES.glob("*.md"))
    assert paths
    for path in paths:
        for n, line in _unfenced_lines(path):
            assert not LINE_CITATION.search(line), f"{path.name}:{n} cites a source line number"


def test_plugin_root_links_resolve():
    files = list(PLUGIN.glob("skills/*/SKILL.md")) + list(REFERENCES.glob("*.md"))
    seen = 0
    for path in files:
        for rel in PLUGIN_ROOT_REF.findall(path.read_text(encoding="utf-8")):
            seen += 1
            assert (PLUGIN / rel).is_file(), f"{path}: {rel} does not exist"
    assert seen


def test_schemas_doc_lists_every_column():
    sys.path.insert(0, str(SCRIPTS))
    import compare_runs
    import diff_reports
    import opa_schema
    import kernel_tests
    import tracy_report
    doc = (REFERENCES / "schemas.md").read_text(encoding="utf-8")
    names = (set(opa_schema.OP_TABLE_BASE) | set(opa_schema.CALL_TRACE_COLUMNS)
             | set(opa_schema.PORT_BLOCKERS_COLUMNS) | set(opa_schema.HOST_OPS_COLUMNS)
             | set(tracy_report.MEASURED_COLUMNS) | set(tracy_report.HOST_FALLBACK_COLUMNS)
             | set(tracy_report.FOOTPRINT_COLUMNS) | set(diff_reports.DIFF_COLUMNS)
             | set(compare_runs.CHANGE_COLUMNS) | set(kernel_tests.KERNEL_TEST_COLUMNS)
             | set(opa_schema.GRAPH_OPS_COLUMNS) | set(opa_schema.GRAPH_HOST_TRANSFERS_COLUMNS)
             | set(opa_schema.CACHE_COMPUTE_COLUMNS) | set(opa_schema.CACHE_DATAFLOW_COLUMNS)
             | set(opa_schema.CACHE_KERNELS_COLUMNS)
             | {"op_table_ids", "static_launches", "static_launches_op_code", "measured_launches_op_code",
                "sources", "graph_launches"})
    missing = sorted(n for n in names if f"`{n}`" not in doc)
    assert not missing, f"schemas.md does not document {missing}"


def test_skill_flow_and_gates():
    text = (PLUGIN / "skills" / "model-op-analysis" / "SKILL.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 120
    assert "static + measured" not in text
    for phrase in ("runs.py list", "measurements.py list", "add a destination"):
        assert phrase in text
    for gate in ("validate_static.py", "same_commit.py", "merge_measured.py", "kernel_tests.py",
                 "kernel_ops.py", "runs.py", "measurements.py", "graph_report.py", "cache_report.py"):
        assert f"<plugin-root>/scripts/{gate}" in text
    assert "`<plugin-root>` is the `tt-model-op-analysis` plugin directory" in text


def test_targets_reference_lists_wormhole_and_quasar_sources():
    text = (PLUGIN / "references" / "targets.md").read_text(encoding="utf-8")
    for profile in ("`n150`", "`n300-1chip`", "`quasar`", "`quasar-bf16`"):
        assert profile in text
    assert "quasar-port" not in text


def test_no_reference_mentions_quasar_port():
    for path in [PLUGIN / "README.md", *(PLUGIN / "references").glob("*.md"),
                 *(PLUGIN / "skills").glob("*/SKILL.md")]:
        assert "quasar-port" not in path.read_text(encoding="utf-8"), path.name


def test_measured_destination_runs_no_whole_model_test():
    text = (PLUGIN / "references" / "measured-destination.md").read_text(encoding="utf-8")
    assert "kernel-op tests" in text and "--target" in text
    assert "Do not run the whole-model test on a destination" in text


def test_root_readme_installs_plugin_on_both_hosts():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "codex plugin add tt-model-op-analysis@tenstorrent-skills" in readme
    assert "/plugin install tt-model-op-analysis@tenstorrent-skills" in readme


def test_validator_accepts_bom_csv(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    for name in ("op_table.csv", "call_trace.csv"):
        path = run / name
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    result = validate(run)
    assert result.returncode == 0, result.stdout


def test_written_csvs_carry_bom_for_spreadsheet_apps(tmp_path):
    src = ops_csv(tmp_path / "ops.csv", [dev("MatmulDeviceOperation", 0, 1, 8, 1)])
    out = tmp_path / "out"
    assert run_script("tracy_report.py", str(src), str(out)).returncode == 0
    for name in ("measured_ops.csv", "host_fallback.csv", "footprint.csv"):
        assert (out / name).read_bytes().startswith(b"\xef\xbb\xbf"), name


def test_kernel_ops_aggregates_by_device_op_and_factory(tmp_path):
    rows = [op_row(1, 2, op_code="HaloDeviceOperation", device_op="prim::halo", program_factory="Halo"),
            op_row(2, 3, op_code="HaloDeviceOperation", device_op="prim::halo", program_factory="Halo",
                   port="❌", **{"quasar-bf16:tests": "t.py::h", "quasar-bf16:test_result": "fail"}),
            op_row(3, 1, op_code="MoveDeviceOperation", device_op="prim::move", program_factory="Move")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 6)])
    m = measured_table(tmp_path / "measured_ops.csv",
                       [("HaloDeviceOperation", "")] * 4 + [("TilizeDeviceOperation", "")])
    result = run_script("kernel_ops.py", str(run), "--measured", str(m))
    assert result.returncode == 0, result.stderr
    got = read(run / "kernel_ops.csv")
    assert [(r["device_op"], r["op_table_ids"], r["static_launches"], r["measured_launches_op_code"])
            for r in got] == [("prim::halo", "1,2", "5", "4"), ("prim::move", "3", "1", "0"),
                              ("", "", "0", "1")]
    assert got[0]["quasar-bf16:statuses"] == "⚠️ 1; ❌ 1"
    assert got[0]["quasar-bf16:test_results"] == "t.py::h: fail"
    assert got[2]["origin"] == "measured" and got[2]["op_code"] == "TilizeDeviceOperation"


def test_kernel_ops_shows_only_results_of_nodes_that_ran(tmp_path):
    tests = {"quasar-bf16:tests": "t.py::ran; t.py::not_run", "quasar-bf16:test_result": "fail"}
    run = make_static_run(tmp_path, [op_row(1, 1, port="❌", **tests)], [trace_row(1, 1, 1)])
    write_rows(run / "kernel_test_results_quasar-bf16.csv",
               [{"id": "1", "node_id": "t.py::ran", "op_table_ids": "1", "result": "fail", "duration_s": "900",
                 "log": "x", "environment": "run.json"}])
    assert run_script("kernel_ops.py", str(run)).returncode == 0
    assert read(run / "kernel_ops.csv")[0]["quasar-bf16:test_results"] == "t.py::ran: fail"


def measured_seq(path, codes):
    return measured_table(path, [(c, "") for c in codes])


def test_merge_marks_static_rows_seen(tmp_path):
    rows = [op_row(1, 1, op_code="A"), op_row(2, 1, op_code="B")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 2, 1)])
    m = measured_seq(tmp_path / "m.csv", ["A"])
    assert run_merge(str(run), str(m)).returncode == 0
    got = read(run / "op_table.csv")
    assert [r["origin"] for r in got] == ["static+measured", "static"]


def test_merge_inserts_measured_only_in_order(tmp_path):
    rows = [op_row(1, 1, op_code="A"), op_row(2, 1, op_code="B")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 2, 1)])
    m = measured_seq(tmp_path / "m.csv", ["A", "X", "X", "B"])
    assert run_merge(str(run), str(m)).returncode == 0
    got = read(run / "op_table.csv")
    assert [(r["id"], r["op_code"], r["origin"], r["p150:launches"]) for r in got] == [
        ("1", "A", "static+measured", "1"), ("2", "X", "measured", "2"), ("3", "B", "static+measured", "1")]
    assert got[1]["call_site"] == "not traced"
    assert got[1]["quasar-bf16:test_result"] == "not run"
    assert validate(run).returncode == 0, validate(run).stdout
    assert json.loads((run / "run.json").read_text())["measured_ops"] == str(m)


def test_merge_is_idempotent(tmp_path):
    rows = [op_row(1, 1, op_code="A")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 1)])
    m = measured_seq(tmp_path / "m.csv", ["A", "X"])
    run_merge(str(run), str(m))
    run_merge(str(run), str(m))
    assert [r["op_code"] for r in read(run / "op_table.csv")] == ["A", "X"]


FAKE_PYTEST = """
import pathlib, sys, time
assert "no:timeout" in sys.argv, "pytest-timeout must be disabled on TTsim"
if "--collect-only" in sys.argv:
    if "--verbosity=-1" not in sys.argv:  # a repo's addopts -v turns the id list into a tree
        print("<Module t.py>")
        sys.exit(0)
    files = [a for a in sys.argv[1:] if a.endswith(".py")]
    for line in pathlib.Path(__file__).with_name("collected.txt").read_text().splitlines():
        # A session with several files loads a conftest that renames every id.
        print(line.replace("[", "[renamed-") if len(files) > 1 else line)
    print("3 tests collected")
    sys.exit(0)
node = sys.argv[-1]
xml = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--junitxml="))
if "sleep" in node:
    time.sleep(30)
marker = pathlib.Path(__file__).with_name("slow_once.done")
if "slow_once" in node and not marker.exists():
    marker.touch()
    time.sleep(30)
if "crash" in node:
    sys.exit(1)
# xfail first: "xfail_c" also contains "fail".
child = {"xfail": "<skipped type='pytest.xfail' message='x'/>", "pass": "", "fail": "<failure message='x'/>"}
kind = next(k for k in child if k in node)
pathlib.Path(xml).write_text(f"<testsuites><testsuite><testcase name='t'>{child[kind]}</testcase></testsuite></testsuites>")
sys.exit(0 if kind != "fail" else 1)
"""


def kernel_test_run(tmp_path, tests_per_row):
    rows = [op_row(i, 1, **{"quasar-bf16:tests": t}) for i, t in enumerate(tests_per_row, 1)]
    run = make_static_run(tmp_path, rows, [trace_row(1, len(rows), 1)])
    data = json.loads((run / "run.json").read_text())
    data["ttsim_environment"] = {"TT_METAL_SIMULATOR": "/x/libttsim.so"}
    (run / "run.json").write_text(json.dumps(data))
    _, sha = git_repo(tmp_path)
    data["tt_metal_commit"] = sha
    (run / "run.json").write_text(json.dumps(data))
    fake = tmp_path / "fake_pytest.py"
    fake.write_text(FAKE_PYTEST)
    nodes = {n.strip() for t in tests_per_row for n in t.split(";") if n.strip()}
    (tmp_path / "collected.txt").write_text("\n".join(sorted(nodes)))
    return run, f"{sys.executable} {fake}"


def test_kernel_tests_plan_dedups_nodes(tmp_path):
    run, _ = kernel_test_run(tmp_path, ["t.py::pass_a", "t.py::pass_a; t.py::fail_b"])
    assert run_script("kernel_tests.py", "plan", str(run)).returncode == 0
    plan = read(run / "kernel_tests_plan_quasar-bf16.csv")
    assert [(p["node_id"], p["op_table_ids"], p["row_statuses"]) for p in plan] == [
        ("t.py::pass_a", "1,2", "⚠️"), ("t.py::fail_b", "2", "⚠️")]


def test_kernel_tests_run_and_apply(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a", "t.py::fail_b", "t.py::xfail_c"])
    run_script("kernel_tests.py", "plan", str(run))
    r = run_script("kernel_tests.py", "run", str(run), "--tt-metal", str(tmp_path / "tt-metal"), "--pytest-cmd", cmd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert [x["result"] for x in read(run / "kernel_test_results_quasar-bf16.csv")] == ["pass", "fail", "xfail"]
    assert run_script("kernel_tests.py", "apply", str(run)).returncode == 0
    got = read(run / "op_table.csv")
    assert [(g["quasar-bf16:test_result"], g["quasar-bf16:status"]) for g in got] == [
        ("pass", "✅"), ("fail", "❌"), ("xfail", "❌")]
    assert got[1]["quasar-bf16:evidence"].startswith("TTsim run: t.py::fail_b fail")
    assert validate(run).returncode == 0, validate(run).stdout


def test_kernel_tests_timeout_decides_nothing(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::sleep_pass", "t.py::pass_b"])
    run_script("kernel_tests.py", "plan", str(run))
    run_script("kernel_tests.py", "run", str(run), "--tt-metal", str(tmp_path / "tt-metal"), "--pytest-cmd", cmd,
               "--timeout", "2")
    assert [x["result"] for x in read(run / "kernel_test_results_quasar-bf16.csv")] == ["timeout", "pass"]
    assert run_script("kernel_tests.py", "apply", str(run)).returncode == 0
    got = read(run / "op_table.csv")
    assert [(g["quasar-bf16:test_result"], g["quasar-bf16:status"]) for g in got] == [("timeout", "⚠️"), ("pass", "✅")]
    assert not got[0]["quasar-bf16:evidence"].startswith("TTsim run:")
    assert validate(run).returncode == 0, validate(run).stdout


def test_kernel_tests_resume_reruns_undecided_nodes(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::slow_once_pass", "t.py::pass_b"])
    run_script("kernel_tests.py", "plan", str(run))
    tm = str(tmp_path / "tt-metal")
    run_script("kernel_tests.py", "run", str(run), "--tt-metal", tm, "--pytest-cmd", cmd, "--timeout", "2")
    assert [x["result"] for x in read(run / "kernel_test_results_quasar-bf16.csv")] == ["timeout", "pass"]
    r = run_script("kernel_tests.py", "run", str(run), "--tt-metal", tm, "--pytest-cmd", cmd)
    assert "1 already done, 1 run" in r.stdout, r.stdout + r.stderr
    assert [x["result"] for x in read(run / "kernel_test_results_quasar-bf16.csv")] == ["pass", "pass"]


def test_kernel_tests_function_id_selects_its_parametrizations(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a"])
    (tmp_path / "collected.txt").write_text("t.py::pass_a[x]\nt.py::pass_a[y]")
    run_script("kernel_tests.py", "plan", str(run))
    r = run_script("kernel_tests.py", "run", str(run), "--tt-metal", str(tmp_path / "tt-metal"), "--pytest-cmd", cmd)
    assert r.returncode == 0, r.stdout + r.stderr


def test_kernel_tests_check_empty_plan(tmp_path):
    run, cmd = kernel_test_run(tmp_path, [""])
    run_script("kernel_tests.py", "plan", str(run))
    r = run_script("kernel_tests.py", "check", str(run), "--tt-metal", str(tmp_path / "tt-metal"), "--pytest-cmd", cmd)
    assert r.returncode == 0 and "0 of 0 plan nodes collected" in r.stdout, r.stdout + r.stderr


def test_result_classes_partition_the_result_values():
    sys.path.insert(0, str(SCRIPTS))
    from opa_schema import FAILING_RESULTS, LEGACY_RESULTS, RESULT_STATUS, TEST_RESULT_VALUES, UNDECIDED_RESULTS
    classes = [set(FAILING_RESULTS), set(UNDECIDED_RESULTS), {"pass"}, {"skip", "not run"}]
    assert sorted(v for c in classes for v in c) == sorted(TEST_RESULT_VALUES)
    assert set(RESULT_STATUS) == set(FAILING_RESULTS) | {"pass"}
    assert set(LEGACY_RESULTS.values()) <= set(TEST_RESULT_VALUES)


def test_row_result_mixed_outcomes():
    sys.path.insert(0, str(SCRIPTS))
    from kernel_tests import _row_result
    assert _row_result(["pass", "timeout"]) == "timeout"
    assert _row_result(["timeout", "error"]) == "error"
    assert _row_result(["timeout", "fail"]) == "fail"
    assert _row_result(["skip", "not run"]) == "not run"


def test_legacy_hang_result_is_an_undecided_timeout(tmp_path):
    run, _ = kernel_test_run(tmp_path, ["t.py::pass_a"])
    rows = read(run / "op_table.csv")
    rows[0].update({"quasar-bf16:status": "❌", "quasar-bf16:test_result": "hang",
                    "quasar-bf16:evidence": "TTsim run: t.py::pass_a hang || fork x"})
    write_rows(run / "op_table.csv", rows)
    assert "is from plugin 0.5.0" in validate(run).stdout
    run_script("kernel_tests.py", "plan", str(run))
    write_rows(run / "kernel_test_results_quasar-bf16.csv",
               [{"id": "1", "node_id": "t.py::pass_a", "op_table_ids": "1", "result": "hang", "duration_s": "900",
                 "log": "x", "environment": "run.json"}])
    r = run_script("kernel_tests.py", "apply", str(run))
    assert "WARNING: row 1" in r.stdout
    assert read(run / "op_table.csv")[0]["quasar-bf16:test_result"] == "timeout"
    items = {i["item"]: i for i in json.loads(run_script("measurements.py", "list", str(run), "--json").stdout)}
    assert items["quasar-bf16:kernel-tests ❌"]["done_nodes"] == 0


def test_kernel_tests_run_refuses_uncollected_node(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a[device_params0]", "t.py::pass_b"])
    (tmp_path / "collected.txt").write_text("t.py::pass_a[quasar-device_params0]\nt.py::pass_b")
    run_script("kernel_tests.py", "plan", str(run))
    r = run_script("kernel_tests.py", "run", str(run), "--tt-metal", str(tmp_path / "tt-metal"), "--pytest-cmd", cmd)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "NOT COLLECTED: t.py::pass_a[device_params0]" in r.stdout
    assert "collected: t.py::pass_a[quasar-device_params0]" in r.stdout
    assert not (run / "kernel_test_results_quasar-bf16.csv").exists()
    only_b = run_script("kernel_tests.py", "run", str(run), "--tt-metal", str(tmp_path / "tt-metal"),
                        "--pytest-cmd", cmd, "--only", "pass_b")
    assert only_b.returncode == 0, only_b.stdout + only_b.stderr
    assert [x["result"] for x in read(run / "kernel_test_results_quasar-bf16.csv")] == ["pass"]


def test_kernel_tests_check_collects_each_file_alone(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a[x]", "u.py::pass_c[y]"])
    run_script("kernel_tests.py", "plan", str(run))
    r = run_script("kernel_tests.py", "check", str(run), "--tt-metal", str(tmp_path / "tt-metal"), "--pytest-cmd", cmd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2 of 2 plan nodes collected" in r.stdout


def test_kernel_tests_check_lists_uncollected_plan_nodes(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a", "u.py::pass_c"])
    (tmp_path / "collected.txt").write_text("t.py::pass_a")
    run_script("kernel_tests.py", "plan", str(run))
    r = run_script("kernel_tests.py", "check", str(run), "--tt-metal", str(tmp_path / "tt-metal"), "--pytest-cmd", cmd)
    assert r.returncode == 3
    assert "NOT COLLECTED: u.py::pass_c" in r.stdout and "1 of 2 plan nodes collected" in r.stdout


def test_kernel_tests_check_fixes_one_added_param_prefix(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a[device_params0]"])
    (tmp_path / "collected.txt").write_text("t.py::pass_a[quasar-device_params0]")
    run_script("kernel_tests.py", "plan", str(run))
    r = run_script("kernel_tests.py", "check", str(run), "--tt-metal", str(tmp_path / "tt-metal"),
                   "--pytest-cmd", cmd, "--fix-node-ids")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "UPDATED: t.py::pass_a[device_params0] -> t.py::pass_a[quasar-device_params0]" in r.stdout
    assert read(run / "op_table.csv")[0]["quasar-bf16:tests"] == "t.py::pass_a[quasar-device_params0]"
    assert read(run / "kernel_tests_plan_quasar-bf16.csv")[0]["node_id"] == "t.py::pass_a[quasar-device_params0]"


def test_kernel_tests_check_does_not_guess_ambiguous_param_prefix(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a[device_params0]"])
    (tmp_path / "collected.txt").write_text(
        "t.py::pass_a[quasar-device_params0]\nt.py::pass_a[wormhole-device_params0]")
    run_script("kernel_tests.py", "plan", str(run))
    r = run_script("kernel_tests.py", "check", str(run), "--tt-metal", str(tmp_path / "tt-metal"),
                   "--pytest-cmd", cmd, "--fix-node-ids")
    assert r.returncode == 3
    assert "NOT COLLECTED: t.py::pass_a[device_params0]" in r.stdout
    assert read(run / "op_table.csv")[0]["quasar-bf16:tests"] == "t.py::pass_a[device_params0]"


def test_kernel_tests_run_resumes(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a"])
    run_script("kernel_tests.py", "plan", str(run))
    run_script("kernel_tests.py", "run", str(run), "--tt-metal", str(tmp_path / "tt-metal"), "--pytest-cmd", cmd)
    out = run_script("kernel_tests.py", "run", str(run), "--tt-metal", str(tmp_path / "tt-metal"), "--pytest-cmd", cmd).stdout
    assert "1 already done, 0 run" in out


def test_kernel_tests_run_needs_environment(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a"])
    data = json.loads((run / "run.json").read_text())
    del data["ttsim_environment"]
    (run / "run.json").write_text(json.dumps(data))
    run_script("kernel_tests.py", "plan", str(run))
    r = run_script("kernel_tests.py", "run", str(run), "--tt-metal", str(tmp_path / "tt-metal"), "--pytest-cmd", cmd)
    assert r.returncode == 2 and "ttsim_environment" in r.stdout


def git_repo(tmp_path):
    repo = tmp_path / "tt-metal"
    repo.mkdir()
    for cmd in (["init", "-q"], ["config", "user.email", "t@e.com"], ["config", "user.name", "t"]):
        subprocess.run(["git", *cmd], cwd=repo, check=True)
    (repo / "a.txt").write_text("1")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "c1"], cwd=repo, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    return repo, sha


def same_commit_run(tmp_path, sha):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"tt_metal_commit": sha}))
    return run


def test_same_commit_ok(tmp_path):
    repo, sha = git_repo(tmp_path)
    r = run_script("same_commit.py", str(same_commit_run(tmp_path, sha)), "--tt-metal", str(repo))
    assert r.returncode == 0, r.stdout


def test_same_commit_rejects_other_sha(tmp_path):
    repo, _ = git_repo(tmp_path)
    r = run_script("same_commit.py", str(same_commit_run(tmp_path, "0" * 40)), "--tt-metal", str(repo))
    assert r.returncode == 1 and "0000000" in r.stdout


def test_same_commit_rejects_local_changes(tmp_path):
    repo, sha = git_repo(tmp_path)
    (repo / "a.txt").write_text("2")
    r = run_script("same_commit.py", str(same_commit_run(tmp_path, sha)), "--tt-metal", str(repo))
    assert r.returncode == 1 and "modified" in r.stdout


def test_sheet_spec_view_keeps_meaning(tmp_path):
    rows = [op_row(1, 1, call_site="models/a1/tt/model.py:1", dispatch_evidence="picks X if (a || b) at x.cpp:1",
                   port_ev="TTsim run: t.py::a hang || Row 1: fork x"),
            op_row(2, 1, call_site="models/a2/tt/model.py:1", dispatch_evidence="picks X if (a || b) at x.cpp:1")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 2, 1)])
    data = json.loads((run / "run.json").read_text())
    data["entry_test_passes"] = {"by": "user", "date": "2026-10-01", "result": "pass"}
    (run / "run.json").write_text(json.dumps(data))
    spec = sheet(run)
    ops = tab(spec, "ttnn calls")
    # different call sites that shorten alike are still shown per row, not reported as constant
    assert "Call site (file:line)" in ops[0]
    info = tab(spec, "Info")
    assert ["Dispatch evidence", "picks X if (a || b) at x.cpp:1"] in info
    assert ["End-to-end test", "✅ passed locally: pass (user, 2026-10-01; build not recorded)"] in info
    row = dict(zip(ops[0], ops[1]))
    assert row["Quasar, BF16 projection: evidence and gap"] == "TTsim run: t.py::a hang\nfork x"


def test_sheet_spec_summary_wrapped_bullets_and_later_tables(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    (run / "summary.md").write_text("# T\n\n| k | v |\n|---|---|\n| skip | me |\n\n## Findings\n\n"
                                    "- Launches: 2 per inference, which\n  wraps here.\n\n| a | b |\n|---|---|\n| x | 1 |\n")
    info = tab(sheet(run), "Info")
    assert ["Launches", "'2 per inference, which wraps here."] in info
    assert ["x", 1] in info and ["skip", "me"] not in info


def test_discover_ttsim_baseline_has_no_commit_specific_workarounds(tmp_path):
    ttsim, tt_metal = tmp_path / "ttsim", tmp_path / "tt-metal"
    (ttsim / "src" / "_out" / "release_qsr").mkdir(parents=True)
    (ttsim / "make.py").write_text("")
    tt_metal.mkdir()
    result = run_script("discover_ttsim.py", "--ttsim-home", str(ttsim), "--tt-metal-home", str(tt_metal))
    assert result.returncode == 0, result.stderr
    quasar = next(a for a in json.loads(result.stdout)["checkouts"][0]["architectures"] if a["arch"] == "quasar")
    environment = quasar["recommended_run"]["environment"]
    assert environment["ARCH_NAME"] == "quasar" and environment["TT_METAL_SLOW_DISPATCH_MODE"] == "1"
    assert "TT_METAL_DISABLE_SFPLOADMACRO" not in environment
    assert "TT_METAL_QUASAR_NOC_API_VERSION" not in environment


def test_kernel_tests_apply_keeps_quoted_or_in_evidence():
    sys.path.insert(0, str(SCRIPTS))
    from opa_schema import split_evidence
    assert split_evidence("picks X if (a || b)") == ("", "picks X if (a || b)")
    assert split_evidence("TTsim run: t hang || picks X if (a || b)") == ("TTsim run: t hang", "picks X if (a || b)")


def test_sheet_spec_kernel_ops_tab_groups_evidence(tmp_path):
    rows = [op_row(1, 1, device_op="prim::a", program_factory="F", op_code="A", port_ev="same"),
            op_row(2, 2, device_op="prim::a", program_factory="F", op_code="A", port_ev="same",
                   stage="s1 m1 conv"),
            op_row(3, 1, device_op="prim::a", program_factory="F", op_code="A", port_ev="other")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 4)])
    assert run_script("kernel_ops.py", str(run)).returncode == 0
    spec = sheet(run)
    assert [t["title"] for t in spec["tabs"]][:4] == ["Info", "Device op support", "Device ops by factory", "ttnn calls"]
    kernel = tab(spec, "Device ops by factory")
    row = dict(zip(kernel[0], kernel[1]))
    assert row["Device op / program factory"] == "prim::a / F"
    assert row["Stages"] == "encoder, s1" and row["Launches (P150)"] == 4
    assert row["Quasar, BF16 projection: evidence and gap"] == "⚠️ rows 1, 2: same\n\n⚠️ row 3: other"
    assert "P150: evidence and gap" not in kernel[0]
    assert kernel[-1][0] == "Total" and kernel[-1][4] == "=SUM(E2:E2)"
    assert spec["expected"]["'Device ops by factory'!E3"] == 4


def test_device_op_support_tab_counts_rows_and_shows_verdicts(tmp_path):
    rows = [op_row(1, 2, device_op="prim::a", program_factory="F", op_code="A", port="❌", port_ev="xfail"),
            op_row(2, 3, device_op="prim::a", program_factory="G", op_code="A", port="⚠️"),
            op_row(3, 1, device_op="prim::b", program_factory="F", op_code="B", port="✅", port_ev="t")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 6)])
    write_support(run, [("prim::a", "✅", "tests/ttnn/unit_tests/quasar/test_a.py::test_other_shape passes")])
    result = run_script("kernel_ops.py", str(run))
    assert result.returncode == 0 and "2 device ops, 1 destination verdicts not checked yet" in result.stdout
    got = read(run / "device_ops.csv")
    assert [(r["device_op"], r["program_factories"], r["op_table_ids"], r["p150:launches"]) for r in got] == [
        ("prim::a", "F; G", "1,2", "5"), ("prim::b", "F", "3", "1")]
    assert [(r["quasar-bf16:rows_ok"], r["quasar-bf16:rows_untested"], r["quasar-bf16:rows_failing"],
             r["quasar-bf16:op_support"]) for r in got] == [("0", "1", "1", "✅"), ("1", "0", "0", "⏳")]
    spec = sheet(run)
    assert [t["title"] for t in spec["tabs"]][:2] == ["Info", "Device op support"]
    support = tab(spec, "Device op support")
    assert "Quasar, BF16 projection: n/a rows" not in support[0]
    first, second = (dict(zip(support[0], r)) for r in support[1:3])
    assert first["Device op"] == "prim::a" and first["Program factories"] == "F\nG"
    assert (first["Quasar, BF16 projection: ✅ rows"], first["Quasar, BF16 projection: ❌ rows"]) == (0, 1)
    assert first["Quasar, BF16 projection: device op supported"] == "✅ supported: quasar/test_a.py::test_other_shape passes"
    assert second["Quasar, BF16 projection: device op supported"] == "⏳ not checked"
    total = dict(zip(support[0], support[-1]))
    assert total["#"] == "Total" and total["Launches (P150)"] == "=SUM(F2:F3)"
    assert spec["expected"]["'Device op support'!F4"] == 6
    assert spec["expected"]["'Device op support'!I4"] == 1


def test_device_op_support_tab_needs_a_destination(tmp_path):
    rows = [op_row(1, 1)]
    for col in ("quasar-bf16:tests", "quasar-bf16:test_result", "quasar-bf16:status", "quasar-bf16:evidence"):
        del rows[0][col]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 1)], destination=None)
    assert run_script("kernel_ops.py", str(run)).returncode == 0
    assert not (run / "device_ops.csv").exists()
    info = tab(sheet(run), "Info")
    start = info.index(["Tabs", "What it holds", "Filled by", "In this sheet"])
    assert {r[0]: r[3] for r in info[start + 1:]}["Device op support"] == "not produced: the run has no destination"


# Final-review fixes


def test_pre_target_run_is_refused_clearly(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    (run / "run.json").write_text(json.dumps({"targets": ["p150", "quasar"]}))
    v = validate(run)
    assert v.returncode == 1 and "predates source/destination targets" in v.stdout


def test_merge_inserts_after_the_occurrence_that_ran(tmp_path):
    rows = [op_row(1, 1, op_code="A", stage="s1 a"), op_row(2, 1, op_code="B", stage="s1 b"),
            op_row(3, 1, op_code="A", stage="s2 a"), op_row(4, 1, op_code="C", stage="s2 c")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 4, 1)])
    m = measured_seq(tmp_path / "m.csv", ["A", "X", "B", "A", "C"])
    run_merge(str(run), str(m))
    assert [r["stage"] for r in read(run / "op_table.csv")] == ["s1 a", "measured only X", "s1 b", "s2 a", "s2 c"]


def test_merge_divides_by_inferences(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1, op_code="A")], [trace_row(1, 1, 1)])
    m = measured_seq(tmp_path / "m.csv", ["A", "X", "A", "X"])
    assert run_merge(str(run), str(m), "--inferences", "2").returncode == 0
    assert read(run / "op_table.csv")[1]["p150:launches"] == "1"
    bad = measured_seq(tmp_path / "m3.csv", ["A", "X", "X", "X"])
    r = run_merge(str(run), str(bad), "--inferences", "2")
    assert r.returncode == 2 and "not divisible" in r.stdout


def no_pass_row(i, code, **kw):
    return op_row(i, 1, op_code=code, **{"p150:status": "⚠️", "p150:evidence": "no passing P150 run recorded"}, **kw)


def test_merge_entry_pass_decides_source_statuses_again(tmp_path):
    rows = [no_pass_row(1, "A"), no_pass_row(2, "M", launch_condition="skipped when the address is reused"),
            no_pass_row(3, "M", launch_condition="skipped when the address is reused"), no_pass_row(4, "B"),
            op_row(5, 1, op_code="C", **{"p150:status": "❌", "p150:evidence": "TT_FATAL at x.cpp:1"})]
    run = make_static_run(tmp_path, rows, [trace_row(1, 5, 1)])
    m = measured_seq(tmp_path / "m.csv", ["A", "M", "C"])
    r = run_merge(str(run), str(m), "--entry-pass", "PCC 0.99", "--date", "2026-10-13", "--build", "Release")
    assert r.returncode == 0, r.stdout
    assert "row 5: p150 status is ❌ but the passing entry test launched C" in r.stdout
    got = {x["id"]: (x["p150:status"], x["p150:evidence"]) for x in read(run / "op_table.csv")}
    assert got["1"] == ("✅", "No CI workflow runs the entry test on P150; it passed on a local P150 under Tracy"
                               " (PCC 0.99, 2026-10-13), launching A 1x per inference.")
    # one of two conditional launches ran; which one is not attributable
    assert got["2"][0] == got["3"][0] == "⚠️" and "1 of 2 potential times" in got["2"][1]
    assert got["4"][0] == "⚠️" and "launched no B" in got["4"][1]
    assert got["5"] == ("❌", "TT_FATAL at x.cpp:1")
    passes = json.loads((run / "run.json").read_text())["entry_test_passes"]
    assert passes == {"by": "source:tracy on a local p150 device", "date": "2026-10-13", "result": "PCC 0.99",
                      "build": "Release"}
    support_from_rows(run)
    v = validate(run)
    assert v.returncode == 0 and "WARNING" not in v.stdout, v.stdout


def test_merge_entry_pass_keeps_ci_record(tmp_path):
    run = make_static_run(tmp_path, [no_pass_row(1, "A")], [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    del data["entry_test_ci"]
    data["entry_test_passes"] = {"ci": "perf.yaml: p150"}
    (run / "run.json").write_text(json.dumps(data))
    run_merge(str(run), str(measured_seq(tmp_path / "m.csv", ["A"])), "--entry-pass", "PCC 0.99", "--build", "Release")
    assert json.loads((run / "run.json").read_text())["entry_test_passes"]["ci"] == "perf.yaml: p150"
    assert "in CI (perf.yaml: p150) and on a local P150" in read(run / "op_table.csv")[0]["p150:evidence"]


def test_merge_without_entry_pass_leaves_statuses_and_validator_warns(tmp_path):
    run = make_static_run(tmp_path, [no_pass_row(1, "A")], [trace_row(1, 1, 1)])
    run_merge(str(run), str(measured_seq(tmp_path / "m.csv", ["A"])))
    assert read(run / "op_table.csv")[0]["p150:status"] == "⚠️"
    support_from_rows(run)
    assert "WARNING" not in validate(run).stdout
    data = json.loads((run / "run.json").read_text())
    data["entry_test_passes"] = {"by": "user", "date": "2026-10-13", "result": "pass"}
    (run / "run.json").write_text(json.dumps(data))
    v = validate(run)
    assert v.returncode == 0 and "rows 1 ran in the measured entry test" in v.stdout, v.stdout


def test_merge_with_blackhole_destination_validates(tmp_path):
    rows = [op_row(1, 1, op_code="A", **{"p100:shapes": "(1,32)", "p100:launches": "1", "p100:status": "✅",
                                        "p100:evidence": "CI"})]
    for col in ("quasar-bf16:tests", "quasar-bf16:test_result",
                "quasar-bf16:status", "quasar-bf16:evidence"):
        del rows[0][col]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 1), trace_row(2, 1, 1, profile="p100")],
                          destination="p100")
    run_merge(str(run), str(measured_seq(tmp_path / "m.csv", ["A", "X"])))
    v = validate(run)
    assert v.returncode == 0, v.stdout


def test_kernel_ops_reads_measured_from_run_json(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1, op_code="A")], [trace_row(1, 1, 1)])
    run_merge(str(run), str(measured_seq(tmp_path / "m.csv", ["A", "A"])))
    run_script("kernel_ops.py", str(run))
    got = read(run / "kernel_ops.csv")[0]
    assert (got["origin"], got["measured_launches_op_code"]) == ("static+measured", "2")


def test_apply_does_not_pass_a_row_with_unrun_tests(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a; t.py::pass_b"])
    write_rows(run / "kernel_test_results_quasar-bf16.csv", [{"id": "1", "node_id": "t.py::pass_a", "op_table_ids": "1",
                                                  "result": "pass", "duration_s": "1", "log": "x"}])
    run_script("kernel_tests.py", "apply", str(run))
    got = read(run / "op_table.csv")[0]
    assert (got["quasar-bf16:test_result"], got["quasar-bf16:status"]) == ("not run", "⚠️")


def test_error_result_does_not_decide_status(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::crash_a"])
    run_script("kernel_tests.py", "plan", str(run))
    r = run_script("kernel_tests.py", "run", str(run), "--tt-metal", str(run.parent / "tt-metal"), "--pytest-cmd", cmd)
    assert r.returncode == 0, r.stdout
    assert read(run / "kernel_test_results_quasar-bf16.csv")[0]["result"] == "error"
    run_script("kernel_tests.py", "apply", str(run))
    got = read(run / "op_table.csv")[0]
    assert (got["quasar-bf16:test_result"], got["quasar-bf16:status"]) == ("error", "⚠️")
    assert validate(run).returncode == 0


def test_kernel_tests_run_refuses_other_commit(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a"])
    data = json.loads((run / "run.json").read_text())
    data["tt_metal_commit"] = "0" * 40
    (run / "run.json").write_text(json.dumps(data))
    run_script("kernel_tests.py", "plan", str(run))
    r = run_script("kernel_tests.py", "run", str(run), "--tt-metal", str(run.parent / "tt-metal"), "--pytest-cmd", cmd)
    assert r.returncode == 1 and "0000000" in r.stdout


def test_kernel_tests_batches_keep_earlier_results(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a", "t.py::fail_b"])
    run_script("kernel_tests.py", "plan", str(run))
    env2 = tmp_path / "env2.json"
    env2.write_text(json.dumps({"TT_METAL_SIMULATOR": "/y/libttsim.so"}))
    tm = str(run.parent / "tt-metal")
    run_script("kernel_tests.py", "run", str(run), "--tt-metal", tm, "--pytest-cmd", cmd, "--only", "pass_a")
    run_script("kernel_tests.py", "run", str(run), "--tt-metal", tm, "--pytest-cmd", cmd, "--only", "fail_b",
               "--env-json", str(env2))
    got = {r["node_id"]: (r["result"], r["environment"]) for r in read(run / "kernel_test_results_quasar-bf16.csv")}
    assert got == {"t.py::pass_a": ("pass", "run.json"), "t.py::fail_b": ("fail", "env2.json")}


def test_node_ids_with_symbols_are_accepted(tmp_path):
    node = "tests/ttnn/x/test_a.py::test_b[H@g0.W@g1-(1, 2)-'x'-*]"
    out = validate(make_static_run(tmp_path, [op_row(1, 1, **{"quasar-bf16:tests": node})],
                                   [trace_row(1, 1, 1)])).stdout
    assert "not a list of pytest node ids" not in out


def test_run_without_destination_validates(tmp_path):
    rows = [op_row(1, 1)]
    for col in ("quasar-bf16:tests", "quasar-bf16:test_result", "quasar-bf16:status", "quasar-bf16:evidence"):
        del rows[0][col]
    result = validate(make_static_run(tmp_path, rows, [trace_row(1, 1, 1)], destination=None))
    assert result.returncode == 0, result.stdout


def test_two_destinations_validate(tmp_path):
    rows = [op_row(1, 2, **{"p100:shapes": "(1,32)", "p100:launches": "1", "p100:status": "✅",
                            "p100:evidence": "CI"})]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 2), trace_row(2, 1, 1, profile="p100")])
    data = json.loads((run / "run.json").read_text())
    data["destinations"] = ["quasar-bf16", "p100"]
    (run / "run.json").write_text(json.dumps(data))
    result = validate(run)
    assert result.returncode == 0, result.stdout
    assert "p100: 1 launches per inference" in result.stdout


def test_wormhole_and_quasar_sources_allowed(tmp_path):
    for source in ("n150", "n300-1chip", "quasar"):
        run = tmp_path / source
        run.mkdir()
        (run / "run.json").write_text(json.dumps({"source": source, "destinations": [], "stage_blocks": ["x"]}))
        assert f"source '{source}' not in" not in validate(run).stdout


def test_bf16_projection_needs_non_quasar_source(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"source": "quasar", "destinations": ["quasar-bf16"], "stage_blocks": ["x"]}))
    assert "quasar-bf16 needs a non-Quasar source" in validate(run).stdout


def test_v03_run_json_is_old_format(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    (run / "run.json").write_text(json.dumps({"source": "p150", "destination": "quasar-bf16"}))
    out = validate(run).stdout
    assert "plugin 0.3.0" in out and "start a new run" in out and "upgrade_run.py" not in out


def to_quasar_source(run, entry="e2e.py::pass_e2e"):
    """Turn a p150 -> quasar-bf16 fixture run into a Quasar-source run without destinations."""
    data = json.loads((run / "run.json").read_text())
    data.update({"source": "quasar", "destinations": [], "test": entry})
    (run / "run.json").write_text(json.dumps(data))
    out = []
    for r in read(run / "op_table.csv"):
        new = {k: v for k, v in r.items() if not k.startswith(("p150:", "quasar-bf16:"))}
        new.update({"quasar:shapes": r["p150:shapes"], "quasar:launches": r["p150:launches"],
                    "quasar:tests": r["quasar-bf16:tests"], "quasar:test_result": r["quasar-bf16:test_result"],
                    "quasar:status": r["quasar-bf16:status"], "quasar:evidence": r["quasar-bf16:evidence"]})
        out.append(new)
    write_rows(run / "op_table.csv", out, list(out[0].keys()))
    trace = read(run / "call_trace.csv")
    for t in trace:
        t["profile"] = "quasar"
    write_rows(run / "call_trace.csv", trace, list(trace[0].keys()))
    return run


def test_quasar_source_without_destination_validates(tmp_path):
    run = to_quasar_source(make_static_run(tmp_path, [op_row(1, 1, **{"quasar-bf16:status": "✅",
                                                                      "quasar-bf16:tests": "t.py::a"})],
                                           [trace_row(1, 1, 1)]))
    result = validate(run)
    assert result.returncode == 0, result.stdout
    assert "quasar: 1 launches per inference" in result.stdout


def test_stage_vocabulary_checked(tmp_path):
    rows = [op_row(1, 1, stage="encoder"), op_row(2, 1, stage="decoder 2 conv")]
    out = validate(make_static_run(tmp_path, rows, [trace_row(1, 2, 1)])).stdout
    assert "row 1: stage 'encoder' needs '<block> <module> <role>'" in out
    assert "row 2: stage block 'decoder' not in run.json stage_blocks" in out


def test_kernel_ops_has_statuses_per_target(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    assert run_script("kernel_ops.py", str(run)).returncode == 0
    header = read(run / "kernel_ops.csv")[0].keys()
    assert {"p150:statuses", "quasar-bf16:statuses", "quasar-bf16:tests"} <= set(header)


def test_add_destination_after_merge(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1, op_code="A")], [trace_row(1, 1, 1)])
    run_merge(str(run), str(measured_seq(tmp_path / "m.csv", ["A", "X"])))
    data = json.loads((run / "run.json").read_text())
    data["destinations"].append("p100")
    (run / "run.json").write_text(json.dumps(data))
    rows = read(run / "op_table.csv")
    for r in rows:
        if r["origin"] == "measured":
            r.update({"p100:shapes": "not traced", "p100:launches": "", "p100:status": "⚠️",
                      "p100:evidence": "measured-only op"})
        else:
            r.update({"p100:shapes": "(1,32)", "p100:launches": "1", "p100:status": "✅", "p100:evidence": "CI"})
    write_rows(run / "op_table.csv", rows, list(rows[0].keys()))
    write_rows(run / "call_trace.csv", [trace_row(1, 1, 1), trace_row(2, 1, 1, profile="p100")])
    result = validate(run)
    assert result.returncode == 0, result.stdout


def test_merge_fills_every_destination(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1, op_code="A", **{"p100:shapes": "(1,32)", "p100:launches": "1",
                                                                  "p100:status": "✅", "p100:evidence": "CI"})],
                          [trace_row(1, 1, 1), trace_row(2, 1, 1, profile="p100")])
    data = json.loads((run / "run.json").read_text())
    data["destinations"] = ["quasar-bf16", "p100"]
    (run / "run.json").write_text(json.dumps(data))
    run_merge(str(run), str(measured_seq(tmp_path / "m.csv", ["A", "X"])))
    added = read(run / "op_table.csv")[1]
    assert (added["p100:shapes"], added["quasar-bf16:test_result"]) == ("not traced", "not run")
    result = validate(run)
    assert result.returncode == 0, result.stdout


def test_sheet_spec_one_tab_per_kernel_test_target(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    write_rows(run / "kernel_test_results_quasar-bf16.csv",
               [{"id": "1", "node_id": "t.py::a", "op_table_ids": "1", "result": "pass", "duration_s": "1",
                 "log": "x", "environment": "run.json"}])
    tabs = [t["title"] for t in sheet(run)["tabs"]]
    assert "TTsim: tests (Quasar, BF16 projection)" in tabs


def test_kernel_tests_status_filter(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a", "t.py::pass_b"])
    rows = read(run / "op_table.csv")
    rows[1]["quasar-bf16:status"] = "❌"
    write_rows(run / "op_table.csv", rows, list(rows[0].keys()))
    run_script("kernel_tests.py", "plan", str(run), "--target", "quasar-bf16")
    tm = str(run.parent / "tt-metal")
    r = run_script("kernel_tests.py", "run", str(run), "--target", "quasar-bf16", "--tt-metal", tm,
                   "--pytest-cmd", cmd, "--status", "❌")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [r["node_id"] for r in read(run / "kernel_test_results_quasar-bf16.csv")] == ["t.py::pass_b"]


def test_kernel_tests_unknown_target_refused(tmp_path):
    run, _ = kernel_test_run(tmp_path, ["t.py::pass_a"])
    r = run_script("kernel_tests.py", "plan", str(run), "--target", "p150")
    assert r.returncode != 0 and "has no kernel-op tests" in (r.stdout + r.stderr)


def test_quasar_source_never_runs_its_entry_test_on_ttsim(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a"])
    to_quasar_source(run)
    (tmp_path / "collected.txt").write_text("e2e.py::pass_e2e\nt.py::pass_a")
    assert run_script("kernel_tests.py", "plan", str(run)).returncode == 0
    assert [p["node_id"] for p in read(run / "kernel_tests_plan_quasar.csv")] == ["t.py::pass_a"]
    tt_metal = str(run.parent / "tt-metal")
    assert run_script("kernel_tests.py", "run", str(run), "--tt-metal", tt_metal, "--pytest-cmd", cmd,
                      "--entry").returncode != 0
    # A plan from an earlier plugin version still lists the entry test; run skips it.
    write_rows(run / "kernel_tests_plan_quasar.csv",
               [{"id": "1", "node_id": "e2e.py::pass_e2e", "op_table_ids": "", "row_statuses": "", "kind": "entry"},
                {"id": "2", "node_id": "t.py::pass_a", "op_table_ids": "1", "row_statuses": "⚠️", "kind": "kernel"}])
    r = run_script("kernel_tests.py", "run", str(run), "--tt-metal", tt_metal, "--pytest-cmd", cmd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert [x["node_id"] for x in read(run / "kernel_test_results_quasar.csv")] == ["t.py::pass_a"]


def test_measurement_items_for_bh_source(tmp_path):
    run, cmd = kernel_test_run(tmp_path, ["t.py::pass_a", "t.py::pass_b"])
    run_script("kernel_tests.py", "plan", str(run), "--target", "quasar-bf16")
    run_script("kernel_tests.py", "run", str(run), "--target", "quasar-bf16", "--tt-metal",
               str(run.parent / "tt-metal"), "--pytest-cmd", cmd, "--only", "pass_a")
    items = json.loads(run_script("measurements.py", "list", str(run), "--json").stdout)
    got = {i["item"]: (i["state"], i["done_nodes"], i["nodes"]) for i in items}
    assert got["source:tracy"] == ("pending", 0, 1)
    assert got["quasar-bf16:kernel-tests ⚠️"] == ("partial", 1, 2)
    assert next(i for i in items if i["item"] == "quasar-bf16:kernel-tests ⚠️")["estimate"].startswith("~")


def test_measurement_estimate_unknown_before_first_test(tmp_path):
    run, _ = kernel_test_run(tmp_path, ["t.py::pass_a"])
    run_script("kernel_tests.py", "plan", str(run))
    items = json.loads(run_script("measurements.py", "list", str(run), "--json").stdout)
    assert items[-1]["estimate"].startswith("unknown")


def test_quasar_source_entry_item(tmp_path):
    run, _ = kernel_test_run(tmp_path, ["t.py::pass_a"])
    to_quasar_source(run)
    run_script("kernel_tests.py", "plan", str(run))
    names = [i["item"] for i in json.loads(run_script("measurements.py", "list", str(run), "--json").stdout)]
    assert names == ["source:kernel-tests ⚠️"]


def test_marked_items_and_pending_filter(tmp_path):
    run, _ = kernel_test_run(tmp_path, ["t.py::pass_a"])
    run_script("kernel_tests.py", "plan", str(run), "--target", "quasar-bf16")
    r = run_script("measurements.py", "mark", str(run), "source:tracy", "failed", "--note", "no p150 device")
    assert r.returncode == 0, r.stdout + r.stderr
    items = {i["item"]: i for i in json.loads(run_script("measurements.py", "list", str(run), "--json").stdout)}
    assert (items["source:tracy"]["state"], items["source:tracy"]["note"]) == ("failed", "no p150 device")
    data = json.loads((run / "run.json").read_text())
    data["measured_ops"] = str(run / "measured_ops.csv")
    (run / "run.json").write_text(json.dumps(data))
    pending = [i["item"] for i in json.loads(run_script("measurements.py", "list", str(run), "--json",
                                                        "--pending").stdout)]
    assert pending == ["source:graph", "source:cache", "quasar-bf16:kernel-tests ⚠️"]
    assert run_script("measurements.py", "mark", str(run), "bogus", "done").returncode == 2


def test_runs_list_separates_implementations(tmp_path):
    for slug in ("ttnn_resnet", "quasar"):
        d = tmp_path / "op-analysis" / "resnet50" / slug / ("p150" if slug == "ttnn_resnet" else "quasar") / "20e872e"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps({"source": d.parent.name, "destinations": [],
                                                "tt_metal_commit": "20e872e3", "date": "2026-10-12",
                                                "implementation": f"models/x/{slug}/tt/m.py", "stage_blocks": ["x"]}))
    old = tmp_path / "op-analysis" / "resnet50" / "static" / "2026-10-12_3ba039a"
    old.mkdir(parents=True)
    (old / "run.json").write_text(json.dumps({"targets": ["p150"]}))
    v03 = tmp_path / "op-analysis" / "resnet50" / "static" / "2026-10-12_20e872e"
    v03.mkdir(parents=True)
    (v03 / "run.json").write_text(json.dumps({"source": "p150", "destination": "quasar-bf16"}))
    r = run_script("runs.py", "list", str(tmp_path), "resnet50", "--json")
    assert r.returncode == 0, r.stdout + r.stderr
    got = {(x["implementation"], x["source"], x["format"]) for x in json.loads(r.stdout)}
    assert got == {("ttnn_resnet", "p150", "current"), ("quasar", "quasar", "current"), ("", "", "old"),
                   ("", "p150", "old")}
    text = run_script("runs.py", "list", str(tmp_path), "resnet50").stdout
    assert "cannot be extended" in text and "ttnn_resnet on p150 @ 20e872e" in text


def test_impl_slug_accepts_module_file_tt_dir_and_implementation_dir():
    sys.path.insert(0, str(SCRIPTS))
    import runs
    base = "models/demos/vision/classification/resnet50"
    assert runs.impl_slug(f"{base}/ttnn_resnet/tt/ttnn_functional_resnet50.py") == "ttnn_resnet"
    assert runs.impl_slug("models/demos/vision/detection/rtdetr/tt") == "rtdetr"
    assert runs.impl_slug(f"{base}/quasar") == "quasar"
    assert runs.impl_slug(f"{base}/blackhole") == "blackhole"


def test_runs_list_setup_only_run_is_not_started(tmp_path):
    run = tmp_path / "op-analysis" / "m" / "impl" / "p150" / "abc1234"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"source": "p150", "implementation": "models/m/impl/tt/m.py",
                                              "tt_metal_commit": "abc1234", "date": "2026-10-22",
                                              "graph_capture": {"path": "g.json", "inferences": 1}}))
    runs = json.loads(run_script("runs.py", "list", str(tmp_path), "m", "--json").stdout)
    assert runs[0]["static"] == "not started"
    assert {m["item"]: m["state"] for m in runs[0]["measurements"]}["source:graph"] == "done"


def test_runs_list_validated_run_is_done(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    target = tmp_path / "op-analysis" / "m" / "impl" / "p150" / "abc1234"
    target.parent.mkdir(parents=True)
    run.rename(target)
    runs = json.loads(run_script("runs.py", "list", str(tmp_path), "m", "--json").stdout)
    assert runs[0]["static"] == "done"


def test_migration_script_is_gone():
    assert not (SCRIPTS / "upgrade_run.py").exists()
    for path in [PLUGIN / "README.md", *(PLUGIN / "references").glob("*.md"),
                 *(PLUGIN / "skills").glob("*/SKILL.md")]:
        assert "upgrade_run" not in path.read_text(encoding="utf-8"), path.name


# Final-review fixes (0.4.0)


def test_measurement_items_without_plan_file(tmp_path):
    run, _ = kernel_test_run(tmp_path, ["t.py::pass_a"])
    to_quasar_source(run)
    assert not (run / "kernel_tests_plan_quasar.csv").exists()
    names = [i["item"] for i in json.loads(run_script("measurements.py", "list", str(run), "--json").stdout)]
    assert names == ["source:kernel-tests ⚠️"]


@pytest.mark.parametrize("undecided", ["error", "timeout"])
def test_undecided_result_does_not_finish_an_item(tmp_path, undecided):
    run, _ = kernel_test_run(tmp_path, ["t.py::pass_a", "t.py::pass_b"])
    run_script("kernel_tests.py", "plan", str(run))
    write_rows(run / "kernel_test_results_quasar-bf16.csv",
               [{"id": "1", "node_id": "t.py::pass_a", "op_table_ids": "1", "result": undecided, "duration_s": "2",
                 "log": "x", "environment": "run.json"},
                {"id": "2", "node_id": "t.py::pass_b", "op_table_ids": "2", "result": "pass", "duration_s": "2",
                 "log": "x", "environment": "run.json"}])
    items = {i["item"]: i for i in json.loads(run_script("measurements.py", "list", str(run), "--json",
                                                         "--pending").stdout)}
    assert (items["quasar-bf16:kernel-tests ⚠️"]["state"], items["quasar-bf16:kernel-tests ⚠️"]["done_nodes"]) == (
        "partial", 1)


def test_resumed_session_offers_only_pending_items():
    skill = (PLUGIN / "skills" / "model-op-analysis" / "SKILL.md").read_text(encoding="utf-8")
    dest = (PLUGIN / "references" / "measured-destination.md").read_text(encoding="utf-8")
    assert "measurements.py list <run dir> --pending" in skill
    assert "measurements.py list <run dir> --pending" in dest


def test_runs_list_survives_unreadable_runs(tmp_path):
    base = tmp_path / "op-analysis" / "m"
    for name, text in (("a", "{not json"), ("b", json.dumps({"source": "p150", "destinations": None,
                                                              "stage_blocks": ["x"]}))):
        d = base / "impl" / "p150" / name
        d.mkdir(parents=True)
        (d / "run.json").write_text(text)
    r = run_script("runs.py", "list", str(tmp_path), "m", "--json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [x["format"] for x in json.loads(r.stdout)] == ["unreadable", "current"]
    assert "Traceback" not in run_script("runs.py", "list", str(tmp_path), "m").stderr


def test_destinations_must_be_a_list(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    (run / "run.json").write_text(json.dumps({"source": "p150", "destinations": "p100", "stage_blocks": ["x"]}))
    r = validate(run)
    assert r.returncode == 1 and "destinations must be a list" in r.stdout and "Traceback" not in r.stderr


def test_probe_device_names_wormhole_profiles(monkeypatch):
    import importlib
    import types
    monkeypatch.setitem(sys.modules, "ttnn", types.ModuleType("ttnn"))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    probe = importlib.import_module("probe_device")
    assert probe._profile("wormhole_b0", {"x": 12, "y": 1}, {"x": 8, "y": 8}) == "n150"
    assert probe._profile("wormhole_b0", {"x": 12, "y": 1}, {"x": 8, "y": 7}) == "n300-1chip"
    assert probe._profile("blackhole", {"x": 7, "y": 1}, {"x": 11, "y": 10}) == "p100"
    assert probe._profile("wormhole_b0", {"x": 12, "y": 1}, {"x": 7, "y": 7}) == "custom"


def test_run_setup_lets_a_continued_run_be_read():
    text = (PLUGIN / "references" / "run-setup.md").read_text(encoding="utf-8")
    assert "the run being continued is the input" in text


def test_n300_is_named_as_one_chip():
    import importlib
    sys.path.insert(0, str(SCRIPTS))
    schema = importlib.import_module("opa_schema")
    assert "n300" not in schema.PROFILES and "n300-1chip" in schema.LAUNCH_PROFILES
    text = (PLUGIN / "references" / "targets.md").read_text(encoding="utf-8")
    assert "`n300`" not in text


def test_upload_skill_asks_root_then_run_then_drive_folder():
    text = (PLUGIN / "skills" / "upload-op-analysis" / "SKILL.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 80
    assert "`<plugin-root>` is the `tt-model-op-analysis` plugin directory" in text
    order = ["**Output root.**", "<plugin-root>/scripts/runs.py list", "<plugin-root>/scripts/validate_static.py",
             "**Drive folder.**", "<plugin-root>/references/drive-upload.md"]
    positions = [text.index(o) for o in order]
    assert positions == sorted(positions)
    assert "drive_url" in text


def test_analysis_skill_hands_upload_to_upload_skill():
    text = (PLUGIN / "skills" / "model-op-analysis" / "SKILL.md").read_text(encoding="utf-8")
    assert "drive-upload.md" not in text
    assert "`upload-op-analysis`" in text


def test_drive_upload_reference_asks_folder_only_without_drive_url():
    text = (PLUGIN / "references" / "drive-upload.md").read_text(encoding="utf-8")
    assert "Ask for the Drive folder only when" in text


def test_runs_list_reports_drive_url(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    data["drive_url"] = "https://docs.google.com/spreadsheets/d/x"
    (run / "run.json").write_text(json.dumps(data))
    target = tmp_path / "op-analysis" / "m" / "impl" / "p150" / "abc1234"
    target.parent.mkdir(parents=True)
    run.rename(target)
    runs = json.loads(run_script("runs.py", "list", str(tmp_path), "m", "--json").stdout)
    assert runs[0]["drive_url"] == "https://docs.google.com/spreadsheets/d/x"
    assert "uploaded: https://docs.google.com/spreadsheets/d/x" in run_script("runs.py", "list", str(tmp_path), "m").stdout


def _skill_step(n):
    text = (PLUGIN / "skills" / "model-op-analysis" / "SKILL.md").read_text(encoding="utf-8")
    start = text.index(f"\n{n}. **")
    end = text.index(f"\n{n + 1}. **") if f"\n{n + 1}. **" in text else text.index("\n## Rules")
    return text[start:end]


def test_skill_asks_output_root_before_listing_runs():
    first = _skill_step(3)
    assert first.index("output root") < first.index("runs.py list")
    assert "output root" not in _skill_step(4)


def test_skill_reprobes_device_before_offering_tracy():
    step = _skill_step(8)
    assert "probe the source device again" in step
    assert "measurements.py mark <run dir> <item> failed" in step
    assert "`source:tracy`, `source:graph` and `source:cache` come from" in step


def test_skill_destination_step_names_quasar_code_modes():
    step = _skill_step(7)
    assert "`quasar`" in step and "`quasar-bf16`" in step


# Graph capture and kernel cache (0.7.0)


def _graph_node(counter, kind, **params):
    return {"connections": [], "counter": counter, "node_type": kind, "params": params}


def graph_capture(path: pathlib.Path, launches: list[tuple], sha: str = "", transfers_at: tuple = (),
                  uploads_at: tuple = ()) -> pathlib.Path:
    """launches: (ttnn api, prim, op code, factory type); transfers_at / uploads_at: launch indices to put
    a to_torch / copy_host_to_device_tensor before."""
    nodes = [_graph_node(0, "capture_start")]

    def add(kind, **params):
        nodes.append(_graph_node(len(nodes), kind, **params))

    def transfer():
        # ttnn.to_torch reads the tensor back (from_device) and then converts it, as two sibling calls.
        add("function_start", name="ttnn.to_torch")
        add("function_start", name="ttnn.from_device")
        add("function_start", name="Tensor::cpu")
        add("function_end", name="Tensor::cpu")
        add("function_end", name="ttnn.from_device")
        add("function_start", name="tt::tt_metal::detail::convert_tt_tensor_to_framework_tensor")
        add("function_end", name="tt::tt_metal::detail::convert_tt_tensor_to_framework_tensor")
        add("function_end", name="ttnn.to_torch")

    def upload():
        add("function_start", name="ttnn.copy_host_to_device_tensor")
        add("function_start", name="tt::tt_metal::copy_to_device")
        add("function_end", name="tt::tt_metal::copy_to_device")
        add("function_end", name="ttnn.copy_host_to_device_tensor")

    transfer()
    for n, (api, prim, code, factory) in enumerate(launches):
        if n in transfers_at:
            transfer()
        if n in uploads_at:
            upload()
        add("function_start", name=api)
        add("function_start", name=prim)
        add("function_start", name=code, program_cache_hit=n > 0, program_factory_index=0,
            program_factory_type=factory)
        for name in (code, prim, api):
            add("function_end", name=name)
    transfer()
    path.write_text(json.dumps({"version": 1, "graph": nodes, "metadata": {"git_sha": sha, "dirty": False}}))
    return path


def test_short_factory_keeps_nested_class_and_templates():
    sys.path.insert(0, str(SCRIPTS))
    import graph_report
    assert graph_report.short_factory("ttnn::operations::pool::Pool2D::MultiCore") == "Pool2D::MultiCore"
    assert graph_report.short_factory("ttnn::prim::ReshardSameWidthFactory<true>") == "ReshardSameWidthFactory<true>"
    assert graph_report.short_factory("ttnn::a::F<ttnn::b::X>") == "F<ttnn::b::X>"


def test_graph_report_counts_launches_per_factory_and_inside_transfers(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    one = [("ttnn.op1", "ttnn::prim::op1", "Op1DeviceOperation", "ttnn::operations::op1::Op1Factory"),
           ("ttnn.foo", "ttnn::prim::foo", "FooDeviceOperation", "ttnn::operations::foo::FooFactory")]
    capture = graph_capture(tmp_path / "g.json", one * 2, transfers_at=(1,), uploads_at=(2,))
    r = run_script("graph_report.py", str(capture), str(run), "--inferences", "2", "--date", "2026-10-21")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = read(run / "graph_ops.csv")
    assert [(x["op_code"], x["program_factory"], x["launches"], x["launches_per_inference"], x["cache_misses"])
            for x in rows] == [("Op1DeviceOperation", "Op1Factory", "2", "1", "1"),
                               ("FooDeviceOperation", "FooFactory", "2", "1", "0")]
    assert rows[0]["device_op"] == "ttnn::prim::op1" and rows[0]["ttnn_apis"] == "ttnn.op1"
    # Transfers between launches count once each, as the outer ttnn call; the leading and trailing
    # to_torch are outside the forward.
    moves = read(run / "graph_host_transfers.csv")
    assert [(m["transfer"], m["after_op_code"], m["before_op_code"], m["count"]) for m in moves] == [
        ("to_torch", "Op1DeviceOperation", "FooDeviceOperation", "1"),
        ("copy_host_to_device", "FooDeviceOperation", "Op1DeviceOperation", "1")]
    recorded = json.loads((run / "run.json").read_text())["graph_capture"]
    assert (recorded["launches"], recorded["kernel_ops"], recorded["host_transfers_inside"]) == (4, 2, 2)


def test_graph_report_refuses_other_commit_and_factoryless_capture(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    data["tt_metal_commit"] = "a" * 40
    (run / "run.json").write_text(json.dumps(data))
    other = graph_capture(tmp_path / "g.json", [("ttnn.op1", "ttnn::prim::op1", "Op1DeviceOperation", "x::F")],
                          sha="b" * 40)
    r = run_script("graph_report.py", str(other), str(run), "--inferences", "1")
    assert r.returncode == 2 and "pinned" in r.stdout and not (run / "graph_ops.csv").exists()
    (tmp_path / "empty.json").write_text(json.dumps({"graph": [_graph_node(0, "capture_start")]}))
    r = run_script("graph_report.py", str(tmp_path / "empty.json"), str(run), "--inferences", "1")
    assert r.returncode == 2 and "program factory" in r.stdout


def test_kernel_ops_adds_graph_launches_and_graph_only_rows(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1, device_op="ttnn::prim::op1")], [trace_row(1, 1, 1)])
    capture = graph_capture(tmp_path / "g.json", [
        ("ttnn.op1", "ttnn::prim::op1", "Op1DeviceOperation", "ttnn::operations::op1::Op1Factory"),
        ("ttnn.foo", "ttnn::prim::foo", "FooDeviceOperation", "ttnn::operations::foo::FooFactory")])
    assert run_script("graph_report.py", str(capture), str(run), "--inferences", "1").returncode == 0
    assert run_script("kernel_ops.py", str(run)).returncode == 0
    rows = read(run / "kernel_ops.csv")
    assert [(r["op_code"], r["origin"], r["graph_launches"], r["sources"]) for r in rows] == [
        ("Op1DeviceOperation", "static", "1", "static, graph"),
        ("FooDeviceOperation", "graph", "1", "graph")]
    kernel = tab(sheet(run), "Device ops by factory")
    header = kernel[0]
    assert "Graph launches per inference" in header and "Seen by" in header
    assert dict(zip(header, kernel[2]))["Device op / program factory"] == "prim::foo / FooFactory"


FAKE_ELFRAY = """#!{python}
import json, pathlib, sys
a = sys.argv[1:]
if "--help" in a:
    print("  --inlined-calls every inlined call"); sys.exit(0)
side = json.loads(pathlib.Path(a[-1] + ".json").read_text())
if a[0] == "--inlined-calls":
    print(json.dumps(side)); sys.exit(0)
print(json.dumps({{"kind": "enum", "name": a[1], "values": [{{"value": 5, "name": "Float16_b"}},
                                                         {{"value": 6, "name": "Bfp8_b"}}]}}))
"""


def _elf(path: pathlib.Path, calls: list[dict], dwarf: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF")
    doc = {"elfray_schema": 2, "kind": "inlined_calls", "machine": "riscv", "dwarf": dwarf,
           "functions": [{"name": "kernel_main", "low_pc": "0x0", "calls": calls}]}
    pathlib.Path(str(path) + ".json").write_text(json.dumps(doc))


def _call(name, depth, template=(), runtime=(), source=None):
    call = {"name": name, "qualified_name": name, "depth": depth, "call_line": 1, "call_column": 1,
            "template_args": [{"name": n, "value": v, "type": None, "enum": None, "display": v} for n, v in template],
            "runtime_args": [{"name": n, "is_static": v is not None, "static_values": v or []} for n, v in runtime]}
    if source:
        call["source_file"] = source
    return call


def kernel_cache(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    engine = tmp_path / "elfray"
    engine.write_text(FAKE_ELFRAY.format(python=sys.executable))
    engine.chmod(0o755)
    kernels = tmp_path / "cache" / "key" / "4097" / "kernels"
    for compile_hash in ("111", "222"):
        build = kernels / "eltwise" / compile_hash
        _elf(build / "trisc0" / "trisc0.elf", [_call("add_tiles", 1, source="hw/inc/api/compute/eltwise.h"),
                                               _call("llk_unpack_AB", 2, [("BType", "NONE")],
                                                     [("operandA", [0]), ("operandB", [1]), ("idx", None)])])
        _elf(build / "trisc1" / "trisc1.elf", [_call("llk_math_eltwise_binary", 1, [("type", "ELWADD")],
                                                     [("dst_index", [0, 1, 2, 5])])])
        _elf(build / "trisc2" / "trisc2.elf", [_call("pack_tile", 1, source="hw/inc/api/compute/pack.h"),
                                               _call("llk_pack", 2, runtime=[("output", [16])])])
        (build / "chlkc_descriptors.h").write_text(
            "constexpr std::int32_t unpack_src_format[32] = {5,6,255};\n"
            "constexpr std::int32_t pack_dst_format[32] = {255,255,255,255,255,255,255,255,255,255,255,255,255,"
            "255,255,255,5};\n"
            "constexpr bool DST_ACCUM_MODE = false;\n#define DST_SYNC_MODE DstSync::SyncHalf\n"
            "constexpr ckernel::MathFidelity MATH_FIDELITY = static_cast<ckernel::MathFidelity>(4);\n")
    _elf(kernels / "reader" / "333" / "brisc" / "brisc.elf",
         [_call("noc_async_read_tile", 1, source="hw/inc/api/dataflow/dataflow_api.h"),
          _call("cb_push_back", 1, runtime=[("cb_id", [0]), ("pages", None)])])
    _elf(kernels / "nodebug" / "444" / "ncrisc" / "ncrisc.elf", [], dwarf=False)
    _elf(tmp_path / "cache" / "key" / "4097" / "firmware" / "trisc0" / "trisc0.elf", [_call("llk_firmware", 1)])
    return tmp_path / "cache", engine


def test_cache_report_builds_elfray_tables(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    cache, engine = kernel_cache(tmp_path)
    r = run_script("cache_report.py", str(cache), str(run), "--elfray", str(engine), "--date", "2026-10-21")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARNING: nodebug 444" in r.stdout and "TT_METAL_RISCV_DEBUG_INFO=1" in r.stdout
    compute = {row["API"]: row for row in read(run / "cache_compute.csv")}
    assert set(compute) == {"llk_unpack_AB<BType=NONE>", "llk_math_eltwise_binary<type=ELWADD>", "llk_pack"}
    unpack = compute["llk_unpack_AB<BType=NONE>"]
    # Two compile hashes of one kernel collapse into one row naming the kernel once.
    assert unpack["Kernels"] == "eltwise" and unpack["TTNN Ops"] == "add_tiles"
    assert unpack["Op Args"] == "operandA=0, operandB=1, idx=?"
    assert unpack["Input Data Formats"] == "cb0=Float16_b, cb1=Bfp8_b" and unpack["Output Data Formats"] == "-"
    assert compute["llk_math_eltwise_binary<type=ELWADD>"]["Op Args"] == "dst_index={0..2,5}"
    assert compute["llk_pack"]["Output Data Formats"] == "cb16=Float16_b"
    assert (unpack["Math Fidelity"], unpack["FP32 Dest Accum"], unpack["Dst Sync Mode"]) == ("HiFi4", "false", "SyncHalf")
    dataflow = [(row["API"], row["Processor(s)"], row["Op Args"]) for row in read(run / "cache_dataflow.csv")]
    assert dataflow == [("cb_push_back", "brisc", "cb_id=0, pages=?"), ("noc_async_read_tile", "brisc", "-")]
    kernels = read(run / "cache_kernels.csv")
    assert sorted(k["kernel"] for k in kernels) == ["eltwise", "eltwise", "nodebug", "reader"]
    recorded = json.loads((run / "run.json").read_text())["kernel_cache"]
    assert (recorded["kernel_builds"], recorded["compute_rows"], recorded["builds_with_errors"]) == (4, 3, 1)


def test_cache_report_without_engine_or_with_bad_layers(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    cache, engine = kernel_cache(tmp_path)
    r = run_script("cache_report.py", str(cache), str(run), "--elfray", str(tmp_path / "missing"))
    assert r.returncode == 3 and "install.sh" in r.stdout
    r = run_script("cache_report.py", str(cache), str(run), "--elfray", str(engine), "--layers", "llk_api,bogus")
    assert r.returncode == 2 and "bogus" in r.stdout
    r = run_script("cache_report.py", str(cache), str(run), "--elfray", str(engine), "--layers", "noc_api")
    assert r.returncode == 2 and "no API call" in r.stdout
    assert "kernel_cache" not in json.loads((run / "run.json").read_text())


def test_source_items_and_confidence_follow_the_methods_run(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import opa_schema
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    items = {i["item"]: i["state"] for i in json.loads(run_script("measurements.py", "list", str(run), "--json").stdout)}
    assert [items[k] for k in ("source:tracy", "source:graph", "source:cache")] == ["pending"] * 3
    info = tab(sheet(run), "Info")
    assert ["Confidence", "low: read from the model code at the pinned commit; nothing was run"] in info
    capture = graph_capture(tmp_path / "g.json", [("ttnn.op1", "ttnn::prim::op1", "Op1DeviceOperation", "x::Op1Factory")])
    assert run_script("graph_report.py", str(capture), str(run), "--inferences", "1").returncode == 0
    items = {i["item"]: i["state"] for i in json.loads(run_script("measurements.py", "list", str(run), "--json").stdout)}
    assert items["source:graph"] == "done" and items["source:cache"] == "pending"
    info = tab(sheet(run), "Info")
    methods = info.index(["Analysis methods", "State"])
    assert [r[0] for r in info[methods + 1:methods + 6]] == [
        *(opa_schema.METHOD_LABELS[m] for m in opa_schema.SOURCE_METHODS), "Confidence"]
    assert info[methods + 3][1].startswith("done") and info[methods + 4][1] == "not run"
    assert info[methods + 5][1].startswith("high:")
    assert opa_schema.confidence(["static", "tracy"]) == "medium"
    assert opa_schema.confidence(["static", "cache"]) == "medium"
    assert run_script("measurements.py", "mark", str(run), "source:cache", "failed", "--note", "no elfray").returncode == 0
    info = tab(sheet(run), "Info")
    assert info[methods + 4][1].startswith("failed") and info[methods + 4][1].endswith("no elfray")


def test_kernel_ops_graph_names_measured_only_rows_and_tracy_is_per_op_code(tmp_path):
    rows = [op_row(1, 1, op_code="MmDeviceOperation", device_op="prim::mm", program_factory="Mm1D",
                   origin="static+measured"),
            op_row(2, 1, op_code="MmDeviceOperation", device_op="prim::mm", program_factory="Mm2D",
                   origin="static+measured"),
            op_row(3, 1, op_code="FooDeviceOperation", device_op="", program_factory="", origin="measured",
                   call_site="not traced", grid_dependency="not traced", stage="measured only FooDeviceOperation")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 2)])
    write_rows(run / "graph_ops.csv", [
        {"id": "1", "op_code": "MmDeviceOperation", "device_op": "ttnn::prim::mm", "program_factory": "Mm1D",
         "program_factory_type": "ttnn::prim::Mm1D", "ttnn_apis": "ttnn.linear", "launches": "1",
         "launches_per_inference": "1", "cache_misses": "1"},
        {"id": "2", "op_code": "FooDeviceOperation", "device_op": "ttnn::prim::foo", "program_factory": "FooF",
         "program_factory_type": "ttnn::foo::FooF", "ttnn_apis": "ttnn.foo", "launches": "1",
         "launches_per_inference": "1", "cache_misses": "1"}])
    sys.path.insert(0, str(SCRIPTS))
    import kernel_ops
    out = kernel_ops.aggregate(read(run / "op_table.csv"), "p150", ["quasar-bf16"],
                               {"MmDeviceOperation": 2, "FooDeviceOperation": 1}, graph=read(run / "graph_ops.csv"))
    got = [(e["op_code"], e["program_factory"], e["origin"], e["graph_launches"], e["sources"]) for e in out]
    assert got == [("MmDeviceOperation", "Mm1D", "static+measured", "1", "static, tracy (op code), graph"),
                   ("MmDeviceOperation", "Mm2D", "static+measured", "0", "static, tracy (op code)"),
                   ("FooDeviceOperation", "FooF", "measured", "1", "tracy, graph")]


def test_graph_capture_plugin_is_inert_without_its_variable():
    text = (PLUGIN / "pytest_plugins" / "opa_graph_capture.py").read_text(encoding="utf-8")
    assert 'ENV = "OPA_GRAPH_CAPTURE"' in text and "if not os.environ.get(ENV)" in text
    source = (PLUGIN / "references" / "measured-source.md").read_text(encoding="utf-8")
    assert "PYTHONPATH=<plugin-root>/pytest_plugins" in source and "-p opa_graph_capture" in source


# Source-method menu and Quasar LLK gap check (0.7.0)


QUASAR_API = """// SPDX
#pragma once
template <BroadcastType BType, bool EN_32BIT_DEST>
inline void llk_unpack_AB(const std::uint32_t operandA, const std::uint32_t operandB) {}

template <bool is_fp32_dest_acc_en, /* note */ MathFidelity math_fidelity>
inline void llk_math_eltwise_binary(std::uint32_t dst_index) {}

inline void llk_math_matmul_tile(const std::uint32_t dst_index) { _llk_math_matmul_tile_(dst_index); }

void caller() { return llk_math_eltwise_binary<false, 4>(0); }
"""
QUASAR_TYPES_H = "enum class DataFormat : std::uint8_t {\n    Float32 = 0,\n    Float16_b = 5, // E8M7\n    Invalid = 0xff\n};\n"


def quasar_repo(tmp_path):
    repo, _ = git_repo(tmp_path)
    api = repo / "tt_metal/hw/ckernels/quasar/metal/llk_api/llk_api.h"
    types = repo / "tt_metal/hw/inc/internal/tt-2xx/quasar/tensix_types.h"
    core = repo / "tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_math_matmul.h"
    for path, text in ((api, QUASAR_API), (types, QUASAR_TYPES_H),
                       (core, "inline void _llk_math_matmul_tile_(const std::uint32_t dst_index) {}\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "quasar"], cwd=repo, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    return repo, sha


def test_llk_gaps_check_and_review(tmp_path):
    repo, sha = quasar_repo(tmp_path)
    compute = tmp_path / "compute.csv"
    # The elfray extension export: no id column, backticks around names.
    write_rows(compute, [
        {"API": "`llk_unpack_AB<BType=NONE, is_fp32_dest_acc_en=false>`", "Kernels": "eltwise",
         "Op Args": "operandA=0, operandB=1", "Input Data Formats": "cb0=Float16_b, cb1=Bfp8_b",
         "Output Data Formats": "-"},
        {"API": "`llk_math_eltwise_binary<is_fp32_dest_acc_en=false, math_fidelity=HiFi4>`", "Kernels": "eltwise",
         "Op Args": "dst_index={0..3}", "Input Data Formats": "-", "Output Data Formats": "cb16=Float16_b"},
        {"API": "`llk_math_matmul<math_fidelity=LoFi>`", "Kernels": "mm", "Op Args": "dst_index=?",
         "Input Data Formats": "-", "Output Data Formats": "-"},
        {"API": "`pack_tile`", "Kernels": "mm", "Op Args": "-", "Input Data Formats": "-", "Output Data Formats": "-"},
    ])
    out = tmp_path / "gaps.csv"
    r = run_script("llk_gaps.py", "check", str(compute), "--tt-metal", str(repo), "--ref", sha, "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    rows = {(x["kind"], x["name"]): x for x in read(out)}
    unpack = rows[("llk_api", "llk_unpack_AB")]
    assert unpack["quasar_status"] == "signature differs"
    assert unpack["missing_template_args"] == "is_fp32_dest_acc_en"
    assert unpack["quasar_template_args"] == "BType, EN_32BIT_DEST"
    assert unpack["quasar_definitions"] == "tt_metal/hw/ckernels/quasar/metal/llk_api/llk_api.h:4"
    # The call inside caller() is not a definition; the commented template still parses.
    binary = rows[("llk_api", "llk_math_eltwise_binary")]
    assert binary["quasar_status"] == "present" and binary["quasar_definitions"].endswith("llk_api.h:7")
    matmul = rows[("llk_api", "llk_math_matmul")]
    assert matmul["quasar_status"] == "missing" and matmul["similar_quasar_apis"] == "llk_math_matmul_tile"
    assert rows[("llk_api", "pack_tile")]["quasar_status"] == "not an LLK"
    assert rows[("data_format", "Bfp8_b")]["quasar_status"] == "missing"
    assert rows[("data_format", "Float16_b")]["quasar_status"] == "present"

    r = run_script("llk_gaps.py", "review", str(out), matmul["id"], "renamed", "--note", "llk_api.h:9 llk_math_matmul_tile")
    assert r.returncode == 0 and "candidates left" in r.stdout
    assert run_script("llk_gaps.py", "review", str(out), binary["id"], "gap", "--note", "x").returncode == 2
    assert run_script("llk_gaps.py", "review", str(out), unpack["id"], "gap", "--note", " ").returncode == 2
    # A new check keeps the verdict.
    assert run_script("llk_gaps.py", "check", str(compute), "--tt-metal", str(repo), "--ref", sha,
                      "--out", str(out)).returncode == 0
    again = {x["name"]: x for x in read(out)}
    assert (again["llk_math_matmul"]["review"], again["llk_math_matmul"]["review_note"]) == (
        "renamed", "llk_api.h:9 llk_math_matmul_tile")


def test_llk_gaps_with_run_dir_records_counts(tmp_path):
    repo, sha = quasar_repo(tmp_path)
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    data["tt_metal_commit"] = sha
    (run / "run.json").write_text(json.dumps(data))
    compute = run / "cache_compute.csv"
    write_rows(compute, [{"id": "1", "API": "llk_math_matmul<math_fidelity=LoFi>", "Kernels": "mm", "Op Args": "-",
                          "Input Data Formats": "-", "Output Data Formats": "-"}])
    r = run_script("llk_gaps.py", "check", str(compute), "--tt-metal", str(repo), "--run-dir", str(run))
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((run / "run.json").read_text())["llk_gaps"]["counts"] == {"llk_api missing": 1}
    assert [t["title"] for t in sheet(run)["tabs"]][-1] == "LLK gaps (Quasar)"


def test_source_methods_lists_all_four_with_reasons(tmp_path):
    repo, sha = git_repo(tmp_path)
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    data["tt_metal_commit"] = sha
    (run / "run.json").write_text(json.dumps(data))

    def menu(*extra):
        r = run_script("source_methods.py", str(run), "--tt-metal", str(repo), "--json", *extra)
        assert r.returncode == 0, r.stdout + r.stderr
        return {m["method"]: m for m in json.loads(r.stdout)}

    got = menu("--device", "none", "--elfray", str(tmp_path / "missing"))
    assert list(got) == ["static", "tracy", "graph", "cache"]
    assert got["static"]["state"] == "done"
    assert all(got[m]["state"] == "unavailable" and "no device" in got[m]["reason"] for m in ("tracy", "graph", "cache"))
    got = menu("--device", "p100", "--elfray", str(tmp_path / "missing"))
    assert "the device is p100" in got["tracy"]["reason"]
    # Right device, but no build and no factory identity at this commit; no engine for the cache.
    got = menu("--device", "p150", "--elfray", str(tmp_path / "missing"))
    assert "no CMake build directory" in got["tracy"]["reason"]
    assert "cannot read ttnn/api/ttnn/graph/graph_consts.hpp" in got["graph"]["reason"]
    assert got["cache"]["state"] == "available, read later" and "install.sh" in got["cache"]["reason"]
    (repo / "build").mkdir()
    (repo / "build" / "CMakeCache.txt").write_text("ENABLE_TRACY:BOOL=ON\n")
    consts = repo / "ttnn/api/ttnn/graph/graph_consts.hpp"
    consts.parent.mkdir(parents=True)
    consts.write_text('constexpr auto kProgramFactoryType = "program_factory_type";\n')
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "graph"], cwd=repo, check=True)
    data["tt_metal_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                             text=True).stdout.strip()
    data["measured_ops"] = "m.csv"
    (run / "run.json").write_text(json.dumps(data))
    _, engine = kernel_cache(tmp_path)
    got = menu("--device", "p150", "--elfray", str(engine))
    assert [got[m]["state"] for m in ("tracy", "graph", "cache")] == ["done", "available", "available"]


def test_skill_offers_the_method_menu():
    step = _skill_step(8)
    assert "<plugin-root>/scripts/source_methods.py" in step and "multi-select" in step
    assert "<plugin-root>/scripts/llk_gaps.py" in step
    source = (PLUGIN / "references" / "measured-source.md").read_text(encoding="utf-8")
    assert "llk_gaps.py review" in source and "does not read DWARF itself" in source


def test_llk_gaps_finds_macro_specified_definitions_and_refuses_commits_without_quasar(tmp_path):
    repo, sha = quasar_repo(tmp_path)
    api = repo / "tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack.h"
    api.write_text("template <bool zero>\nTT_ALWAYS_INLINE void llk_pack_relu_config(const std::uint32_t config) {}\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "pack"], cwd=repo, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    compute = tmp_path / "compute.csv"
    write_rows(compute, [{"API": "llk_pack_relu_config", "Kernels": "k", "Op Args": "config=0",
                          "Input Data Formats": "cb0=Float32, cb1=Float32", "Output Data Formats": "cb16=Float32"}])
    out = tmp_path / "gaps.csv"
    assert run_script("llk_gaps.py", "check", str(compute), "--tt-metal", str(repo), "--ref", head,
                      "--out", str(out)).returncode == 0
    rows = {x["name"]: x for x in read(out)}
    assert rows["llk_pack_relu_config"]["quasar_status"] == "present"
    # One compute row counts once per format, however many CBs use it.
    assert rows["Float32"]["cache_rows"] == "1"
    (tmp_path / "old").mkdir()
    _, old = git_repo(tmp_path / "old")
    r = run_script("llk_gaps.py", "check", str(compute), "--tt-metal", str(tmp_path / "old" / "tt-metal"),
                   "--ref", old, "--out", str(out))
    assert r.returncode == 2 and "cannot be checked" in r.stdout


def test_llk_gaps_drops_a_verdict_when_the_finding_changes(tmp_path):
    repo, sha = quasar_repo(tmp_path)
    compute = tmp_path / "compute.csv"
    write_rows(compute, [{"API": "llk_unpack_AB<BType=NONE, is_fp32_dest_acc_en=false>", "Kernels": "k",
                          "Op Args": "-", "Input Data Formats": "-", "Output Data Formats": "-"}])
    out = tmp_path / "gaps.csv"
    run_script("llk_gaps.py", "check", str(compute), "--tt-metal", str(repo), "--ref", sha, "--out", str(out))
    row_id = read(out)[0]["id"]
    assert run_script("llk_gaps.py", "review", str(out), row_id, "renamed", "--note", "EN_32BIT_DEST").returncode == 0
    api = repo / "tt_metal/hw/ckernels/quasar/metal/llk_api/llk_api.h"
    api.write_text("\n" + api.read_text())
    subprocess.run(["git", "commit", "-qam", "move"], cwd=repo, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    run_script("llk_gaps.py", "check", str(compute), "--tt-metal", str(repo), "--ref", head, "--out", str(out))
    assert read(out)[0]["review"] == ""


def test_graph_entry_pass_decides_source_statuses(tmp_path):
    rows = [op_row(1, 1, op_code="Op1DeviceOperation", **{"p150:status": "⚠️", "p150:evidence": "no passing run"}),
            op_row(2, 1, op_code="Op2DeviceOperation", **{"p150:status": "⚠️", "p150:evidence": "no passing run"})]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 2)])
    capture = graph_capture(tmp_path / "g.json", [("ttnn.op1", "ttnn::prim::op1", "Op1DeviceOperation", "x::Op1Factory")])
    r = run_script("graph_report.py", str(capture), str(run), "--inferences", "1", "--entry-pass", "PCC 0.99", "--build", "Release",
                   "--date", "2026-10-22")
    assert r.returncode == 0, r.stdout + r.stderr
    ops = read(run / "op_table.csv")
    assert [o["p150:status"] for o in ops] == ["✅", "⚠️"]
    assert "under a graph capture (PCC 0.99, 2026-10-22)" in ops[0]["p150:evidence"]
    passes = json.loads((run / "run.json").read_text())["entry_test_passes"]
    assert passes["by"] == "source:graph on a local p150 device"
    assert validate(run).returncode == 0


def test_source_methods_reads_an_existing_cache_without_a_device(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    data["measurements"] = {"source:cache": {"state": "failed", "date": "2026-10-21", "note": "no DWARF"}}
    (run / "run.json").write_text(json.dumps(data))
    _, engine = kernel_cache(tmp_path)
    r = run_script("source_methods.py", str(run), "--device", "none", "--elfray", str(engine), "--json")
    cache = next(m for m in json.loads(r.stdout) if m["method"] == "cache")
    assert cache["state"] == "read an existing cache" and "failed earlier: no DWARF" in cache["reason"]


# Info bring-up checks, tab list, header notes and readable evidence (0.8.0)


def info_rows(spec):
    """Info label -> value; the first occurrence, since the tab list repeats some tab names as labels."""
    out = {}
    for r in tab(spec, "Info"):
        if len(r) > 1 and r[0]:
            out.setdefault(r[0], r[1])
    return out


def test_info_bring_up_checks(tmp_path):
    rows = [op_row(1, 3, grid_dependency="8x4 CoreGrid, models/a/tt/model.py:12"),
            op_row(2, 1, grid_dependency="automatic"),
            op_row(3, 2, grid_dependency="shard spec at models/a/tt/model.py:12 and models/a/tt/model.py:40")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 6)])
    data = json.loads((run / "run.json").read_text())
    data.update({"tracy_mode": "live", "profile_values": {"arch": "blackhole", "compute_grid": "13x10"},
                 "configurable_layers": {"value": "yes", "evidence": "MODEL_NUM_LAYERS read at models/a/tt/model.py:5",
                                         "setting": "MODEL_NUM_LAYERS"},
                 "entry_test_passes": {"by": "source:tracy on a local p150 device", "date": "2026-10-20",
                                       "result": "PCC 0.99"},
                 "graph_capture": {"inferences": 2, "host_transfers_inside": 3, "peak_dram_mib": 812.5,
                                   "peak_dram_mib_per_device": {"0": 812.5}}})
    (run / "run.json").write_text(json.dumps(data))
    write_rows(run / "host_fallback.csv", [{"scope": "model", "window": "warm (signposts a..b)", "device_ops": "95",
                                           "host_ops": "5", "device_op_time_ms": "1", "host_op_time_ms": "1",
                                           "total_time_ms": "10", "host_pct_of_total": "4.20"}])
    write_rows(run / "host_ops.csv", [{"id": "1", "host_work": "torch.cat", "when": "per inference",
                                      "counts_as_host_op": "yes", "reason": "in forward"}], HOST_FIELDS)
    spec = sheet(run)
    info = info_rows(spec)
    assert info["End-to-end test"] == "✅ passed locally: PCC 0.99 (source:tracy on a local p150 device, 2026-10-20; build not recorded)"
    host = info["Host fallback"].lstrip("'")
    assert host.startswith("⚠️ 5 of 100 ops on the host (5.0%), 4.20% of the time (Tracy live")
    assert info["Test hardware"] == "P150: arch blackhole, compute grid 13x10"
    assert info["Peak DRAM"].lstrip("'").startswith("812.5 MiB allocated at once")
    grid = info["Blockers to a smaller grid"]
    assert grid.startswith("❌ 2 of 3 ttnn call rows (5 launches per inference)") and "set at 2 places" in grid
    assert info["Configurable grid"].startswith("✅ yes")
    assert info["Configurable layer count"].startswith("✅ yes: MODEL_NUM_LAYERS")
    values = tab(spec, "Info")
    notes = {values[u["range"]["startRowIndex"]][0]: u["rows"][0]["values"][0]["note"]
             for r in spec["requests"] if (u := r.get("updateCells")) and u["range"]["sheetId"] == 0
             and u["range"]["endRowIndex"] - u["range"]["startRowIndex"] == 1 and u["range"]["startColumnIndex"] == 0
             and u["range"]["endColumnIndex"] == 1}
    assert "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE" in notes["Configurable grid"]
    assert notes["Configurable layer count"].startswith("Setting: MODEL_NUM_LAYERS")
    for name in ("Static trace of the model code", "Tracy profile of the entry test",
                 "ttnn graph capture of the entry test", "Kernel cache of the entry test (elfray)"):
        assert "What it does:" in notes[name] and "Confidence:" in notes[name]
    assert notes["Confidence"].startswith("Confidence in the kernel-op list")
    del data["configurable_layers"]
    (run / "run.json").write_text(json.dumps(data))
    data["configurable_grid"] = {"value": "no", "evidence": "8x4 CoreGrid at models/a/tt/model.py:12"}
    (run / "run.json").write_text(json.dumps(data))
    spec = sheet(run)
    assert info_rows(spec)["Configurable layer count"] == "⏳ not checked"
    assert info_rows(spec)["Configurable grid"].startswith("❌ no")
    del data["graph_capture"]["peak_dram_mib"]
    (run / "run.json").write_text(json.dumps(data))
    assert info_rows(sheet(run))["Peak DRAM"] == "⏳ not measured; a graph capture records it"
    grid_note = next(u["rows"][0]["values"][0]["note"] for r in spec["requests"]
                     if (u := r.get("updateCells")) and "TT_METAL_CORE_GRID_OVERRIDE" in str(u["rows"]))
    assert "need a code change" in grid_note
    layer_note = next(u["rows"][0]["values"][0]["note"] for r in spec["requests"]
                      if (u := r.get("updateCells")) and "_NUM_LAYERS" in str(u["rows"]))
    assert "<MODEL>_NUM_LAYERS" in layer_note and "not checked yet" in layer_note
    # A later local failure wins over an earlier pass.
    data["entry_test_failure"] = {"by": "source:graph on a local p150 device", "date": "2026-10-21",
                                  "result": "PCC 0.71 < 0.90"}
    (run / "run.json").write_text(json.dumps(data))
    assert info_rows(sheet(run))["End-to-end test"].startswith("❌ failed: PCC 0.71 < 0.90")


def test_info_lists_every_tab_and_headers_carry_notes(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    assert run_script("kernel_ops.py", str(run)).returncode == 0
    spec = sheet(run)
    info = tab(spec, "Info")
    start = info.index(["Tabs", "What it holds", "Filled by", "In this sheet"])
    listed = {r[0]: r[3] for r in info[start + 1:]}
    assert all(listed[t["title"]] == "yes" for t in spec["tabs"])
    # Tabs this run cannot have yet are listed with the reason.
    assert listed["Tracy: ops"] == "not produced: the tracy profile of the entry test has not run"
    assert listed["Graph: launches"].startswith("not produced: the ttnn graph capture")
    assert listed["TTsim: tests (Quasar, BF16 projection)"] == "not produced: the TTsim kernel-op tests have not run"
    noted = {r["updateCells"]["range"]["sheetId"]: [v.get("note", "") for v in r["updateCells"]["rows"][0]["values"]]
             for r in spec["requests"] if "updateCells" in r}
    for t in spec["tabs"][1:]:
        header = t["values"][0]
        assert len(noted[t["sheetId"]]) == len(header), t["title"]
        assert all(noted[t["sheetId"]]), (t["title"], [h for h, n in zip(header, noted[t["sheetId"]]) if not n])


def test_kernel_ops_evidence_orders_by_status_and_writes_short_repeats_once(tmp_path):
    rows = [op_row(1, 1, device_op="prim::a", program_factory="F", op_code="A",
                   **{"quasar-bf16:status": "✅", "quasar-bf16:evidence": "fork F runs test_a.py:3; bf16 no-op."}),
            op_row(2, 1, device_op="prim::a", program_factory="F", op_code="A",
                   **{"quasar-bf16:status": "❌", "quasar-bf16:evidence": "fork F hangs, test_b.py:9 xfail; bf16 no-op."}),
            op_row(3, 1, device_op="prim::a", program_factory="F", op_code="A",
                   **{"quasar-bf16:status": "⚠️", "quasar-bf16:evidence": "no test covers fork F with this input"})]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 3)])
    assert run_script("kernel_ops.py", str(run)).returncode == 0
    kernel = tab(sheet(run), "Device ops by factory")
    evidence = dict(zip(kernel[0], kernel[1]))["Quasar, BF16 projection: evidence and gap"]
    assert evidence.split("\n\n") == [
        "❌ row 2: fork F hangs, test_b.py:9 xfail.",
        "⚠️ row 3: no test covers fork F with this input",
        "✅ row 1: fork F runs test_a.py:3.",
        "Rows 1, 2: bf16 no-op."]


def test_kernel_op_tests_are_grouped_by_file():
    sys.path.insert(0, str(SCRIPTS))
    import sheet_spec
    assert sheet_spec.grouped_nodes("a/t1.py::x[1]; a/t1.py::y; b/t2.py::z: pass") == (
        "a/t1.py\n  ::x[1]\n  ::y\nb/t2.py\n  ::z: pass")


def test_summary_bullets_put_labels_in_their_own_column(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    (run / "summary.md").write_text("# M\n\n## Gaps\n\n- **Confirmed gaps (14)**, all in one kernel.\n"
                                    "- Renamed (22): mostly arguments.\n- Not a gap (58), wrappers only.\n")
    info = tab(sheet(run), "Info")
    assert ["Confirmed gaps (14)", "all in one kernel."] in info
    assert ["Renamed (22)", "mostly arguments."] in info
    assert ["Not a gap (58)", "wrappers only."] in info


def test_graph_report_records_peak_dram_and_entry_failure(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    capture = graph_capture(tmp_path / "g.json", [("ttnn.op1", "ttnn::prim::op1", "Op1DeviceOperation", "x::Op1Factory")])
    doc = json.loads(capture.read_text())
    mib = 2**20
    for kind, address, size in (("buffer_allocate", 1, 300 * mib), ("buffer_allocate", 2, 200 * mib),
                                ("buffer_deallocate", 1, 300 * mib), ("buffer_allocate", 3, 50 * mib),
                                ("buffer_deallocate", 99, 999 * mib)):
        doc["graph"].append({"counter": len(doc["graph"]), "node_type": kind, "connections": [],
                             "params": {"type": "DRAM", "device_id": "0", "address": str(address), "size": str(size)}})
    doc["graph"].append({"counter": len(doc["graph"]), "node_type": "buffer_allocate", "connections": [],
                         "params": {"type": "L1", "device_id": "0", "address": "7", "size": str(900 * mib)}})
    capture.write_text(json.dumps(doc))
    r = run_script("graph_report.py", str(capture), str(run), "--inferences", "1", "--entry-fail", "PCC 0.5", "--build", "Release",
                   "--date", "2026-10-22")
    assert r.returncode == 0, r.stdout + r.stderr
    data = json.loads((run / "run.json").read_text())
    assert data["graph_capture"]["peak_dram_mib"] == 500.0
    assert data["entry_test_failure"] == {"by": "source:graph on a local p150 device", "date": "2026-10-22",
                                          "result": "PCC 0.5", "build": "Release"}
    assert read(run / "op_table.csv")[0]["p150:status"] == "✅"
    assert run_script("graph_report.py", str(capture), str(run), "--inferences", "1", "--entry-fail", "x",
                      "--build", "Release", "--entry-pass", "y").returncode == 2


def test_status_legend_moves_to_count_notes(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1), op_row(2, 2, port="❌")], [trace_row(1, 1, 3)])
    (run / "summary.md").write_text(
        "# M\n\n## Status legend\n\n- ✅: P150 - the entry test passed. Quasar BF16 - a Quasar test runs it.\n"
        "- ❌: Quasar BF16 - directly blocked.\n\n## Findings\n\n- **Launches**: 3.\n")
    spec = sheet(run)
    values = tab(spec, "Info")
    assert ["Status legend"] not in values and ["Findings"] in values
    notes = {values[u["range"]["startRowIndex"]][0]: u["rows"][0]["values"][0]["note"]
             for r in spec["requests"] if (u := r.get("updateCells")) and u["range"]["sheetId"] == 0
             and u["range"]["endRowIndex"] - u["range"]["startRowIndex"] == 1}
    assert notes["P150: ✅"] == "The entry test passed."
    assert notes["Quasar, BF16 projection: ⚠️"] == "The code path exists but nothing tests it."
    assert notes["Quasar, BF16 projection: ❌"] == "Directly blocked."



def test_legend_note_picks_the_closest_target():
    sys.path.insert(0, str(SCRIPTS))
    from sheet_spec import legend_note
    labels = ["P150", "Quasar", "Quasar, BF16 projection"]
    text = "P150 - passed. Ops with fixed grids - none. Quasar - runs on Quasar. Quasar BF16 - runs with BF16."
    assert legend_note(text, "Quasar, BF16 projection", labels) == "Runs with BF16."
    assert legend_note(text, "Quasar", labels) == "Runs on Quasar."
    assert legend_note(text, "P150", labels) == "Passed. Ops with fixed grids - none."
    assert legend_note("Quasar BF16 - directly blocked.", "P150", labels) is None
    assert legend_note("blocked everywhere.", "P150", labels) == "blocked everywhere."


def test_validator_checks_configurability_fields(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    data["configurable_grid"] = {"value": "maybe", "evidence": ""}
    data["configurable_layers"] = {"value": "yes", "evidence": "NUM_LAYERS read at m.py:3"}
    (run / "run.json").write_text(json.dumps(data))
    r = validate(run)
    assert r.returncode == 1 and "configurable_grid must be" in r.stdout
    assert "configurable_layers needs \"setting\"" in r.stdout
    del data["configurable_layers"]
    (run / "run.json").write_text(json.dumps(data))
    assert "run.json has no configurable_layers" in validate(run).stdout


def test_destination_question_lists_plain_quasar():
    step = _skill_step(7)
    assert "`quasar` (the code as\n   written, on Quasar)" in step or "`quasar` (the code as written" in step
    assert "`quasar-bf16`" in step


def test_review_fixes_for_sheet_text():
    sys.path.insert(0, str(SCRIPTS))
    import sheet_spec
    assert sheet_spec.grouped_nodes("tests/a/test_x.py; b/t.py::z: pass") == "tests/a/test_x.py\nb/t.py\n  ::z: pass"
    members = [{"id": "1", "q:status": "❌", "q:evidence": "Uses G, e.g. on matmul. Hangs test_b.py:9."},
               {"id": "2", "q:status": "✅", "q:evidence": "Uses F, e.g. on matmul. Passes test_a.py:3."}]
    assert sheet_spec.grouped_evidence(members, "q:evidence") == (
        "❌ row 1: Uses G, e.g. on matmul. Hangs test_b.py:9.\n\n✅ row 2: Uses F, e.g. on matmul. Passes test_a.py:3.")
    members = [{"id": "1", "q:status": "⚠️", "q:evidence": ""},
               {"id": "2", "q:status": "✅", "q:evidence": "runs t.py:1; bf16 no-op"},
               {"id": "3", "q:status": "✅", "q:evidence": "runs t.py:2; bf16 no-op"}]
    assert sheet_spec.grouped_evidence(members, "q:evidence").startswith("⚠️ row 1: no evidence written.")


def test_bold_emphasis_is_not_a_label(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    (run / "summary.md").write_text("# M\n\n## F\n\n- **Not** supported on Quasar yet.\n- **Note:** this one.\n")
    info = tab(sheet(run), "Info")
    assert ["", "Not supported on Quasar yet."] in info
    assert ["Note", "this one."] in info


def test_untraced_rows_keep_grid_blockers_unknown_and_ci_pass_stays_visible(tmp_path):
    rows = [op_row(1, 1), op_row(2, 1, origin="measured", call_site="not traced", grid_dependency="not traced",
                                     stage="measured only Op2DeviceOperation")]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    del data["entry_test_ci"]
    data["entry_test_passes"] = {"ci": "w.yml: job"}
    data["entry_test_failure"] = {"by": "source:graph on a local p150 device", "date": "2026-10-22", "result": "bad"}
    (run / "run.json").write_text(json.dumps(data))
    info = info_rows(sheet(run))
    assert info["Blockers to a smaller grid"].startswith("⚠️ none in the static trace; 1 measured-only rows")
    assert info["End-to-end test"] == "❌ failed: bad (source:graph on a local p150 device, 2026-10-22; build not recorded); CI: w.yml: job"


def test_failure_after_a_local_pass_is_flagged(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    data["entry_test_passes"] = {"by": "source:tracy on a local p150 device", "date": "2026-10-20", "result": "ok"}
    (run / "run.json").write_text(json.dumps(data))
    capture = graph_capture(tmp_path / "g.json", [("ttnn.op1", "ttnn::prim::op1", "Op1DeviceOperation", "x::Op1Factory")])
    r = run_script("graph_report.py", str(capture), str(run), "--inferences", "1", "--entry-fail", "bad", "--build", "Release",
                   "--date", "2026-10-21")
    assert r.returncode == 0 and "superseded run" in r.stdout
    assert "superseded run" in validate(run).stdout


def test_every_llk_gaps_and_cache_column_has_a_note():
    sys.path.insert(0, str(SCRIPTS))
    import opa_schema
    import sheet_notes
    for title, columns in (("LLK gaps (Quasar)", opa_schema.LLK_GAPS_COLUMNS),
                           ("Cache: compute APIs", opa_schema.CACHE_COMPUTE_COLUMNS),
                           ("Cache: dataflow APIs", opa_schema.CACHE_DATAFLOW_COLUMNS),
                           ("Cache: kernels", opa_schema.CACHE_KERNELS_COLUMNS),
                           ("Graph: launches", opa_schema.GRAPH_OPS_COLUMNS),
                           ("Graph host transfers", opa_schema.GRAPH_HOST_TRANSFERS_COLUMNS)):
        assert all(sheet_notes.column_note(title, c) for c in columns), (title, [c for c in columns
                                                                                 if not sheet_notes.column_note(title, c)])


def test_abbreviations_do_not_split_evidence():
    sys.path.insert(0, str(SCRIPTS))
    import sheet_spec
    assert sheet_spec._clauses("passes at 2x3, incl. DRAM TILE input; bf16.") == ["passes at 2x3, incl. DRAM TILE input", "bf16"]


def test_factory_choice_is_shown_in_plain_words(tmp_path):
    rows = [op_row(1, 1, confidence="verified"), op_row(2, 1, confidence="unverified")]
    spec = sheet(make_static_run(tmp_path, rows, [trace_row(1, 1, 2)]))
    ops = tab(spec, "ttnn calls")
    column = ops[0].index("Factory choice")
    assert [r[column] for r in ops[1:]] == ["fixed by the code", "decided at run time"]


def test_large_tabs_stay_in_the_run_directory_and_long_attributes_are_cut(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    write_rows(run / "measured_ops.csv", [{"id": str(i), "op_code": "Op1DeviceOperation", "op_type": "tt_dnn_device",
                                          "attributes": "x" * 400, "inputs": "", "outputs": "", "core_count": "1",
                                          "device_kernel_ns": "1", "host_ns": "1"} for i in range(1, 4)])
    spec = sheet(run)
    ops = tab(spec, "Tracy: ops")
    assert ops[1][3].endswith(" … (cut; whole in measured_ops.csv)") and len(ops[1][3]) < 400
    spec = sheet(run, "--max-tab-kb", "0")
    assert "Tracy: ops" not in [t["title"] for t in spec["tabs"]]
    info = tab(spec, "Info")
    row = next(r for r in info if r and r[0] == "Tracy: ops")
    assert row[3].startswith("in the run directory only: measured_ops.csv, 3 rows")


def test_llk_gap_review_carries_an_icon(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    base = {k: "" for k in ("template_args", "missing_template_args", "quasar_template_args", "missing_runtime_args",
                            "similar_quasar_apis", "quasar_definitions", "configs", "cache_rows", "kernels", "review_note")}
    write_rows(run / "llk_gaps_quasar.csv", [
        base | {"id": "1", "kind": "llk_api", "name": "a", "quasar_status": "missing", "review": "gap"},
        base | {"id": "2", "kind": "llk_api", "name": "b", "quasar_status": "missing", "review": "renamed"},
        base | {"id": "3", "kind": "llk_api", "name": "c", "quasar_status": "signature differs", "review": ""},
        base | {"id": "4", "kind": "llk_api", "name": "d", "quasar_status": "present", "review": ""}],
        ["id", "kind", "name", "quasar_status", "review", "review_note", "template_args", "missing_template_args",
         "quasar_template_args", "missing_runtime_args", "similar_quasar_apis", "quasar_definitions", "configs",
         "cache_rows", "kernels"])
    gaps = tab(sheet(run), "LLK gaps (Quasar)")
    assert [r[4] for r in gaps[1:]] == ["❌ gap", "✅ renamed", "⏳ not reviewed", ""]


def test_host_work_flag_carries_an_icon(tmp_path):
    host = [{"id": "1", "host_work": "torch.cat", "when": "per inference", "counts_as_host_op": "yes", "reason": "r"},
            {"id": "2", "host_work": "weights", "when": "once", "counts_as_host_op": "no", "reason": "r"}]
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)], host_ops=host)
    rows = tab(sheet(run), "Host work in code")
    assert [r[3] for r in rows[1:]] == ["⚠️ yes", "✅ no"]


@pytest.mark.parametrize("device_ops,host_ops,code_steps,mark", [
    (100, 0, 0, "✅"), (100, 0, 1, "✅"), (95, 5, 0, "⚠️"), (90, 10, 0, "❌")])
def test_host_fallback_icon(tmp_path, device_ops, host_ops, code_steps, mark):
    host = [{"id": str(i), "host_work": "w", "when": "per inference", "counts_as_host_op": "yes", "reason": "r"}
            for i in range(1, code_steps + 1)]
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)], host_ops=host)
    data = json.loads((run / "run.json").read_text())
    data["tracy_mode"] = "live"
    (run / "run.json").write_text(json.dumps(data))
    write_rows(run / "host_fallback.csv", [{"scope": "model", "window": "warm", "device_ops": str(device_ops),
                                           "host_ops": str(host_ops), "device_op_time_ms": "1", "host_op_time_ms": "0",
                                           "total_time_ms": "1", "host_pct_of_total": "0.00"}])
    assert info_rows(sheet(run))["Host fallback"].lstrip("'").startswith(mark + " ")


def test_info_counts_llk_gaps(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    base = {k: "" for k in ("review_note", "template_args", "missing_template_args", "quasar_template_args",
                            "missing_runtime_args", "similar_quasar_apis", "quasar_definitions", "configs",
                            "cache_rows", "kernels")}
    columns = ["id", "kind", "name", "quasar_status", "review", "review_note", "template_args", "missing_template_args",
               "quasar_template_args", "missing_runtime_args", "similar_quasar_apis", "quasar_definitions", "configs",
               "cache_rows", "kernels"]
    write_rows(run / "llk_gaps_quasar.csv", [
        base | {"id": "1", "kind": "llk_api", "name": "a", "quasar_status": "present", "review": ""},
        base | {"id": "2", "kind": "llk_api", "name": "b", "quasar_status": "missing", "review": "gap"},
        base | {"id": "3", "kind": "llk_api", "name": "c", "quasar_status": "signature differs", "review": ""},
        base | {"id": "4", "kind": "data_format", "name": "Bfp8_b", "quasar_status": "missing", "review": ""}], columns)
    spec = sheet(run)
    info = tab(spec, "Info")
    start = info.index(["LLK gaps (Quasar)", "Count"])
    got = {r[0].strip(): spec["expected"][f"Info!B{start + 2 + i}"] for i, r in enumerate(info[start + 1:start + 10])}
    assert got == {"LLK APIs in the cache": 3, "found on Quasar by name": 1, "signature differs": 1,
                   "not found by name": 1, "Reviewed: ❌ gap": 1, "Reviewed: ✅ renamed": 0,
                   "Reviewed: ✅ not a gap": 0, "⏳ not reviewed yet": 1, "Data formats missing on Quasar": 1}
    assert info[start + 1][1].startswith("=COUNTIF('LLK gaps (Quasar)'!B2:B5")
    # Without the tab in the sheet the counts are values, not formulas.
    spec = sheet(run, "--max-tab-kb", "0")
    info = tab(spec, "Info")
    start = info.index(["LLK gaps (Quasar)", "Count"])
    assert info[start + 1][1] == 3


def test_host_fallback_needs_a_live_tracy_run(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    write_rows(run / "host_fallback.csv", [{"scope": "model", "window": "warm", "device_ops": "100", "host_ops": "0",
                                           "device_op_time_ms": "1", "host_op_time_ms": "0", "total_time_ms": "1",
                                           "host_pct_of_total": "0.00"}])
    assert info_rows(sheet(run))["Host fallback"].startswith("not decided")
    data = json.loads((run / "run.json").read_text())
    for mode, start in (("trace", "⚠️ not measured"), ("live", "✅ 0 of 100 ops on the host")):
        data["tracy_mode"] = mode
        (run / "run.json").write_text(json.dumps(data))
        assert info_rows(sheet(run))["Host fallback"].lstrip("'").startswith(start)
    m = measured_seq(tmp_path / "m.csv", ["Op1DeviceOperation"])
    assert run_script("merge_measured.py", str(run), str(m)).returncode == 2


# Entry test preflight and CI coverage (0.9.0)


def test_validator_wants_entry_test_ci(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    del data["entry_test_ci"]
    (run / "run.json").write_text(json.dumps(data))
    r = validate(run)
    assert r.returncode == 0 and "run.json has no entry_test_ci" in r.stdout
    data["entry_test_ci"] = {"value": "maybe"}
    (run / "run.json").write_text(json.dumps(data))
    r = validate(run)
    assert r.returncode != 0 and "entry_test_ci must be" in r.stdout


@pytest.mark.parametrize("ci, shown", [
    ({"value": "yes", "evidence": ".github/workflows/perf.yaml: p150"}, "✅ yes on P150: .github/workflows/perf.yaml: p150"),
    ({"value": "no", "evidence": "only on n150"}, "⚠️ no on P150, so any pass below is from a local run: only on n150"),
    ({"value": "n/a", "evidence": "never run"}, "n/a on P150: never run"),
    (None, "⏳ not checked"),
    ("yes", "⏳ not checked"),
])
def test_info_shows_entry_test_ci_below_entry_test(tmp_path, ci, shown):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    data.update({"test": "t.py::e2e", "entry_test_ci": ci})
    (run / "run.json").write_text(json.dumps(data))
    info = tab(sheet(run), "Info")
    labels = [r[0] for r in info if r]
    assert labels.index("Entry test in CI") == labels.index("Entry test") + 1
    assert next(r for r in info if r and r[0] == "Entry test in CI")[1] == shown


def test_legacy_ci_pass_record_shows_as_ci(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    data = json.loads((run / "run.json").read_text())
    del data["entry_test_ci"]
    data.update({"test": "t.py::e2e", "entry_test_passes": {"ci": "w.yml: job"}})
    (run / "run.json").write_text(json.dumps(data))
    info = tab(sheet(run), "Info")
    assert next(r for r in info if r and r[0] == "Entry test in CI")[1] == "✅ yes on P150: w.yml: job"


def test_graph_decide_uses_recorded_capture_and_ci(tmp_path):
    rows = [op_row(1, 1, op_code="Op1DeviceOperation", **{"p150:status": "⚠️", "p150:evidence": "no passing run"}),
            op_row(2, 1, op_code="Op2DeviceOperation", **{"p150:status": "⚠️", "p150:evidence": "no passing run"})]
    run = make_static_run(tmp_path, rows, [trace_row(1, 1, 2)])
    data = json.loads((run / "run.json").read_text())
    data["entry_test_ci"] = {"value": "yes", "evidence": "perf.yaml: p150"}
    (run / "run.json").write_text(json.dumps(data))
    ops_before = (run / "op_table.csv").read_text()
    (run / "op_table.csv").rename(run / "op_table.keep")
    capture = graph_capture(tmp_path / "g.json", [("ttnn.op1", "ttnn::prim::op1", "Op1DeviceOperation", "x::Op1Factory")])
    # Preflight: before the static phase there is no op_table.csv, so the pass decides nothing yet.
    r = run_script("graph_report.py", str(capture), str(run), "--inferences", "1", "--entry-pass", "PCC 0.99", "--build", "Release",
                   "--date", "2026-10-22")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (run / "op_table.csv").exists()
    (run / "op_table.keep").rename(run / "op_table.csv")
    assert (run / "op_table.csv").read_text() == ops_before
    capture.unlink()
    r = run_script("graph_report.py", str(run), "--decide")
    assert r.returncode == 0, r.stdout + r.stderr
    ops = read(run / "op_table.csv")
    assert [o["p150:status"] for o in ops] == ["✅", "⚠️"]
    assert ops[0]["p150:evidence"].startswith("in CI (perf.yaml: p150) and on a local P150 under a graph capture")


def test_graph_decide_needs_a_recorded_pass(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    r = run_script("graph_report.py", str(run), "--decide")
    assert r.returncode == 2 and "--decide needs" in r.stdout
    assert run_script("graph_report.py", str(run)).returncode == 2


def test_preflight_is_documented_and_ttsim_runs_kernel_ops_only():
    setup = (PLUGIN / "references" / "run-setup.md").read_text()
    assert "## Entry test preflight" in setup and "Never for a Quasar source" in setup
    flow = (PLUGIN / "skills" / "model-op-analysis" / "SKILL.md").read_text()
    heads = ["**New run.**", "**Entry test preflight.**", "**Static.**"]
    assert [flow.index(h) for h in heads] == sorted(flow.index(h) for h in heads)
    assert "never an entry test or any whole-model test" in flow
    for path in [PLUGIN / "README.md", *(PLUGIN / "references").glob("*.md"), *SCRIPTS.glob("*.py")]:
        assert "ttsim-e2e" not in path.read_text(encoding="utf-8"), path.name


def test_entry_pass_records_build_and_open_issue_downgrades_it(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1, op_code="Op1DeviceOperation")], [trace_row(1, 1, 1)])
    capture = graph_capture(tmp_path / "g.json", [("ttnn.op1", "ttnn::prim::op1", "Op1DeviceOperation", "x::Op1Factory")])
    args = [str(capture), str(run), "--inferences", "1", "--entry-pass", "PCC 0.99", "--date", "2026-10-23"]
    assert "need --build" in run_script("graph_report.py", *args).stdout
    assert run_script("graph_report.py", *args, "--build", "Release").returncode == 0
    assert json.loads((run / "run.json").read_text())["entry_test_passes"]["build"] == "Release"
    info = info_rows(sheet(run))
    assert info["End-to-end test"].startswith("✅ passed locally: PCC 0.99")
    assert info["End-to-end test"].endswith("Release build, TT_ASSERT off)")
    assert info["Entry test issues"] == "✅ none open (searched 2026-10-01)"
    data = json.loads((run / "run.json").read_text())
    data["entry_test_issues"] = {"checked": "2026-10-24", "open": [
        {"url": "https://github.com/tenstorrent/tt-metal/issues/61620", "title": "conv bias/weight dtype mismatch",
         "effect": "TT_ASSERT fails at startup in assertion-enabled builds"}]}
    (run / "run.json").write_text(json.dumps(data))
    info = info_rows(sheet(run))
    assert info["End-to-end test"].startswith("⚠️ passed locally: PCC 0.99")
    assert "open #61620 reports it failing" in info["End-to-end test"]
    assert info["Entry test issues"].startswith("⚠️ 1 open (searched 2026-10-24): #61620 conv bias/weight dtype")


def test_analysed_by_on_info_and_validated(tmp_path):
    run = make_static_run(tmp_path, [op_row(1, 1)], [trace_row(1, 1, 1)])
    support_from_rows(run)
    data = json.loads((run / "run.json").read_text())
    data["analysed_by"].append({"agent": "Codex", "model": "gpt-x", "work": "source:graph", "date": "2026-10-23"})
    (run / "run.json").write_text(json.dumps(data))
    assert info_rows(sheet(run))["Analysed by"] == ("Claude Code (m): static trace, 2026-10-01; "
                                                   "Codex (gpt-x): source:graph, 2026-10-23")
    assert validate(run).returncode == 0
    data["analysed_by"] = [{"agent": "Codex"}]
    data["entry_test_passes"] = {"by": "user", "date": "2026-10-01", "result": "pass", "build": "Fast"}
    (run / "run.json").write_text(json.dumps(data))
    out = validate(run).stdout
    assert "analysed_by must be a list of" in out and "build 'Fast' not in" in out
    del data["analysed_by"], data["entry_test_issues"]
    data["entry_test_passes"]["build"] = ""
    (run / "run.json").write_text(json.dumps(data))
    out = validate(run).stdout
    assert "has no analysed_by" in out and "has no entry_test_issues" in out and "entry_test_passes has no build" in out
