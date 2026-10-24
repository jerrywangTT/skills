"""Gate for a model-op-analysis run: schema, row ids, status values, call sites and launch totals.

Prints `OK` and the launch totals per target when the run passes. Warnings (evidence a reader
cannot scan: row-number prefixes, overlong text) are printed but do not fail the gate.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from opa_schema import (  # noqa: E402
    ALWAYS, BUILD_TYPES, CALL_SITE, CALL_TRACE_COLUMNS, CI_KEY, CI_VALUES, CONFIDENCE_VALUES, CONFIG_KEYS, CONFIG_VALUES,
    GRID_KEYWORDS,
    latest_entry_failure, HOST_OP_VALUES, HOST_OPS_COLUMNS, LAUNCH_PROFILES,
    LEGACY_RESULTS, NOT_CHECKED, NOT_TRACED, OP_SUPPORT_VALUES, ORIGIN_VALUES, PORT_BLOCKERS_COLUMNS, RESULT_STATUS, ROW_PREFIX,
    SOURCE_TARGETS, STATUS_VALUES, TARGETS, TEST_RESULT_VALUES, TEST_TARGETS, device_op_support_columns, op_key,
    op_table_columns, read_csv, read_header, run_targets, split_evidence,
)

# Evidence is the decisive fact and its <file>:<line>; longer text is a walkthrough, not evidence.
MAX_EVIDENCE_CHARS = 300
# A pytest node id: path, then ::-separated names, with an optional [parametrize id] that may hold
# any character except the ';' this column uses as a separator.
NODE_ID = re.compile(r"^[\w./-]+\.py(::\w+)*(\[[^;]*\])?$")


def _int(value: str, where: str, errors: list[str]) -> int | None:
    try:
        n = int(value)
    except ValueError:
        errors.append(f"{where}: '{value}' is not an integer")
        return None
    if n < 0:
        errors.append(f"{where}: {n} is negative")
    return n


def _targets(run: dict, errors: list[str]) -> tuple[str, list[str]]:
    if "targets" in run and "source" not in run:
        errors.append("run.json predates source/destination targets (plugin 0.2.0); start a new static run")
        return "", []
    if "destination" in run:
        errors.append("run.json has a single 'destination' (plugin 0.3.0); this format cannot be "
                      "extended; start a new run")
        return "", []
    if "source" not in run:
        errors.append("run.json: needs 'source' and 'destinations'")
        return "", []
    if not isinstance(run.get("destinations", []), list):
        errors.append("run.json: destinations must be a list of target names")
        return "", []
    source, destinations = run_targets(run)
    if source not in SOURCE_TARGETS:
        errors.append(f"run.json: source '{source}' not in {SOURCE_TARGETS}")
    for destination in destinations:
        if destination not in TARGETS:
            errors.append(f"run.json: destination '{destination}' not in {TARGETS}")
        if destination == source:
            errors.append(f"run.json: destination '{destination}' equals the source")
        if destination == "quasar-bf16" and source == "quasar":
            errors.append("run.json: quasar-bf16 needs a non-Quasar source")
    if len(set(destinations)) != len(destinations):
        errors.append("run.json: destinations repeat")
    if not isinstance(run.get("stage_blocks"), list) or not run["stage_blocks"]:
        errors.append("run.json: stage_blocks must list the model's block names")
    return source, destinations


def _check_ids(name: str, rows: list[dict], errors: list[str]) -> None:
    for n, row in enumerate(rows, 1):
        if row["id"] != str(n):
            errors.append(f"{name} row {n} has id {row['id']}")


def _check_device_op_support(path: pathlib.Path, ops: list[dict], destinations: list[str], errors: list[str],
                             warnings: list[str]) -> None:
    """Per-device-op verdicts (target-status.md, "Device op support") against the row statuses."""
    name = path.name
    if not path.exists():
        warnings.append(f"{name} missing; the Device op support tab shows every destination as not checked "
                        "(target-status.md, \"Device op support\")")
        return
    header = read_header(path)
    required = device_op_support_columns(destinations)
    bad = [f"{name} missing column {c}" for c in required if c not in header]
    bad += [f"{name} unexpected column {c}" for c in header if c not in required]
    if bad:
        errors.extend(bad)
        return
    members: dict[str, list[dict]] = {}
    for r in ops:
        members.setdefault(op_key(r), []).append(r)
    seen = set()
    unchecked = []
    for n, row in enumerate(read_csv(path), 1):
        where = f"{name} row {n}"
        if None in row or None in row.values():
            errors.append(f"{where}: field count differs from the header")
            continue
        op = row["device_op"]
        if op not in members:
            errors.append(f"{where}: device_op '{op}' is in no op_table row (device_op, or op_code when it is empty)")
            continue
        if op in seen:
            errors.append(f"{where}: device_op '{op}' repeats")
        seen.add(op)
        for t in destinations:
            verdict, evidence = row[f"{t}:op_support"], row[f"{t}:op_evidence"].strip()
            statuses = {m[f"{t}:status"] for m in members[op]}
            if verdict in ("", NOT_CHECKED):
                unchecked.append(f"{op} on {t}")
                continue
            if verdict not in OP_SUPPORT_VALUES:
                errors.append(f"{where}: {t}:op_support '{verdict}' not in {OP_SUPPORT_VALUES}")
                continue
            if not evidence:
                errors.append(f"{where}: {t}:op_support without {t}:op_evidence")
            elif len(evidence) > MAX_EVIDENCE_CHARS:
                warnings.append(f"{where}: {t}:op_evidence exceeds {MAX_EVIDENCE_CHARS} characters")
            if (verdict == "n/a") != (statuses == {"n/a"}):
                errors.append(f"{where}: {t}:op_support '{verdict}' with row statuses {sorted(statuses)};"
                              " n/a exactly when every row of the op is n/a")
            elif verdict in ("⚠️", "❌") and "✅" in statuses:
                errors.append(f"{where}: {t}:op_support '{verdict}' but a row of the op is ✅, so the destination"
                              " runs it; the verdict is ✅")
            elif verdict == "❌" and statuses - {"❌", "n/a"}:
                # A check that fails whatever the arguments fails every call; a ⚠️ row has a factory for its call.
                errors.append(f"{where}: {t}:op_support '❌' but rows of the op are {sorted(statuses - {'❌', 'n/a'})};"
                              " a failure for any arguments makes every row ❌")
    missing = [op for op in members if op not in seen]
    if missing:
        warnings.append(f"{name} has no row for {', '.join(missing)}; the sheet shows them as not checked")
    if unchecked:
        warnings.append(f"{name}: no verdict for {', '.join(unchecked)}; the sheet shows them as not checked")


def validate(run_dir: pathlib.Path) -> tuple[list[str], list[str], dict[str, tuple[int, int, int]]]:
    errors: list[str] = []
    warnings: list[str] = []
    totals: dict[str, tuple[int, int, int]] = {}
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    source, destinations = _targets(run, errors)
    if errors:
        return errors, warnings, totals
    targets = (source, *destinations)
    launch_targets = [source] + [d for d in destinations if d in LAUNCH_PROFILES]

    tables = {
        "op_table.csv": op_table_columns(source, destinations),
        "call_trace.csv": list(CALL_TRACE_COLUMNS),
        "port_blockers.csv": list(PORT_BLOCKERS_COLUMNS),
        "host_ops.csv": list(HOST_OPS_COLUMNS),
    }
    for name, required in tables.items():
        path = run_dir / name
        if not path.exists():
            errors.append(f"{name} missing")
            continue
        header = read_header(path)
        missing = [col for col in required if col not in header]
        errors += [f"{name} missing column {col}" for col in missing]
        errors += [f"{name} unexpected column {col}" for col in header if col not in required]
    if errors:
        return errors, warnings, totals

    rows = {name: read_csv(run_dir / name) for name in tables}
    for name, table in rows.items():
        for n, row in enumerate(table, 1):
            if None in row or None in row.values():
                errors.append(f"{name} row {n}: field count differs from the header")
    if errors:
        return errors, warnings, totals
    for name, table in rows.items():
        _check_ids(name, table, errors)
    ops = rows["op_table.csv"]
    # A measured-only row is traced on the source alone; other targets carry no launch count.
    launches = {t: [None if r["origin"] == "measured" and t != source
                    else _int(r[f"{t}:launches"], f"op_table.csv row {n} {t}:launches", errors)
                    for n, r in enumerate(ops, 1)] for t in launch_targets}

    for n, row in enumerate(ops, 1):
        where = f"op_table.csv row {n}"
        measured_only = row["origin"] == "measured"
        if row["origin"] not in ORIGIN_VALUES:
            errors.append(f"{where}: origin '{row['origin']}' not in {ORIGIN_VALUES}")
        if row["confidence"] not in CONFIDENCE_VALUES:
            errors.append(f"{where}: confidence '{row['confidence']}' not in {CONFIDENCE_VALUES}")
        if measured_only:
            for col in ("call_site", "grid_dependency"):
                if row[col].strip() != NOT_TRACED:
                    errors.append(f"{where}: measured-only row needs {col} '{NOT_TRACED}'")
        elif not CALL_SITE.search(row["call_site"]):
            errors.append(f"{where}: call_site '{row['call_site']}' has no <file>:<line>")
        condition = row["launch_condition"].strip()
        if not condition:
            errors.append(f"{where}: launch_condition empty (use '{ALWAYS}' or name the condition)")
        elif condition != ALWAYS and condition.lower().startswith(ALWAYS):
            errors.append(f"{where}: launch_condition '{condition}' must be exactly '{ALWAYS}' or a condition")
        grid = row["grid_dependency"].strip()
        if not measured_only and grid not in GRID_KEYWORDS and not CALL_SITE.search(grid):
            errors.append(f"{where}: grid_dependency '{grid}' is not {GRID_KEYWORDS} or a value with <file>:<line>")
        # The sheet counts every value other than a bare keyword as a hardcoded grid.
        elif not measured_only and grid not in GRID_KEYWORDS and (lead := re.match(r"\w+", grid)) \
                and lead.group().lower() in GRID_KEYWORDS:
            errors.append(f"{where}: grid_dependency '{grid}' must be exactly '{lead.group().lower()}'; "
                          "put the explanation in dispatch_evidence")
        if not measured_only:
            tokens = row["stage"].split()
            if len(tokens) < 2:
                errors.append(f"{where}: stage '{row['stage']}' needs '<block> <module> <role>'")
            elif tokens[0] not in run["stage_blocks"]:
                errors.append(f"{where}: stage block '{tokens[0]}' not in run.json stage_blocks")
        for target in targets:
            status = row[f"{target}:status"]
            if status not in STATUS_VALUES:
                errors.append(f"{where}: {target}:status '{status}' not in {STATUS_VALUES}")
            if not row[f"{target}:evidence"].strip():
                errors.append(f"{where}: {target}:status without {target}:evidence")
            count = launches[target][n - 1] if target in launch_targets else None
            if count is not None and (count == 0) != (status == "n/a"):
                errors.append(f"{where}: {target} has {count} launches with status '{status}';"
                              " n/a exactly when the profile launches nothing")
            if target in TEST_TARGETS:
                cell = row[f"{target}:tests"]
                if any(not NODE_ID.match(t) for t in (x.strip() for x in cell.split(";")) if t):
                    errors.append(f"{where}: {target}:tests '{cell}' is not a list of pytest node ids")
                result = row[f"{target}:test_result"].strip()
                if result in LEGACY_RESULTS:
                    errors.append(f"{where}: {target}:test_result '{result}' is from plugin 0.5.0; rerun"
                                  " kernel_tests.py apply and withdraw the result (measured-destination.md)")
                elif result not in TEST_RESULT_VALUES:
                    errors.append(f"{where}: {target}:test_result '{result}' not in {TEST_RESULT_VALUES}")
                elif result in RESULT_STATUS and status != RESULT_STATUS[result]:
                    errors.append(f"{where}: {target}:test_result '{result}' requires status {RESULT_STATUS[result]}")

    for column in ["dispatch_evidence"] + [f"{t}:evidence" for t in targets]:
        # A TTsim result prepended by kernel_tests.py is not part of the written evidence.
        written = {r["id"]: split_evidence(r[column])[1] for r in ops}
        prefixed = [i for i, text in written.items() if ROW_PREFIX.match(text)]
        if prefixed:
            warnings.append(f"op_table.csv {column}: rows {', '.join(prefixed)} start with 'Row <n>'; "
                            "rows with one signature share one evidence text")
        long = [i for i, text in written.items() if len(text) > MAX_EVIDENCE_CHARS]
        if long:
            warnings.append(f"op_table.csv {column}: rows {', '.join(long)} exceed {MAX_EVIDENCE_CHARS} characters; "
                            "keep the decisive fact and its <file>:<line>")

    for key in CONFIG_KEYS:
        entry = run.get(key)
        if entry is None:
            warnings.append(f"run.json has no {key}; record it (static-phase.md, step 6)")
        elif not isinstance(entry, dict) or entry.get("value") not in CONFIG_VALUES or not str(entry.get("evidence", "")).strip():
            errors.append(f"run.json {key} must be {{\"value\": one of {CONFIG_VALUES}, \"evidence\": \"...\"}}")
    ci = run.get(CI_KEY)
    if ci is None:
        warnings.append(f"run.json has no {CI_KEY}; record it (run-setup.md, setup order)")
    elif not isinstance(ci, dict) or ci.get("value") not in CI_VALUES or not str(ci.get("evidence", "")).strip():
        errors.append(f"run.json {CI_KEY} must be {{\"value\": one of {CI_VALUES}, \"evidence\": \"...\"}}")
    for key in ("entry_test_passes", "entry_test_failure"):
        record = run.get(key) or {}
        if record.get("by"):
            build = record.get("build", "")
            if not build:
                warnings.append(f"run.json {key} has no build; record the tt-metal build type it ran on "
                                "(measured-source.md, step 1)")
            elif build not in BUILD_TYPES:
                errors.append(f"run.json {key} build '{build}' not in {BUILD_TYPES}")
    for item, record in (run.get("measurements") or {}).items():
        if item.startswith("source:") and record.get("state") == "failed" and "cause" not in record:
            warnings.append(f"run.json measurements {item} failed without a cause; mark it again with --cause "
                            "test, tool or not-run (measured-source.md)")
    issues = run.get("entry_test_issues")
    if issues is None:
        warnings.append("run.json has no entry_test_issues; search the open issues of the entry test "
                        "(run-setup.md, setup order)")
    elif (not isinstance(issues, dict) or not str(issues.get("checked", "")).strip()
          or not isinstance(issues.get("open"), list)
          or any(not isinstance(i, dict) or not all(str(i.get(f, "")).strip() for f in ("url", "title", "effect"))
                 for i in issues["open"])):
        errors.append("run.json entry_test_issues must be {\"checked\": \"YYYY-MM-DD\", \"open\": "
                      "[{\"url\", \"title\", \"effect\"}, ...]}")
    agents = run.get("analysed_by")
    if not agents:
        warnings.append("run.json has no analysed_by; record the agent that did this work (SKILL.md, Rules)")
    elif not isinstance(agents, list) or any(
            not isinstance(a, dict) or not all(str(a.get(f, "")).strip() for f in ("agent", "model", "work", "date"))
            for a in agents):
        errors.append("run.json analysed_by must be a list of {\"agent\", \"model\", \"work\", \"date\"}")
    layers = run.get("configurable_layers")
    if isinstance(layers, dict) and layers.get("value") in ("yes", "partial") and not str(layers.get("setting", "")).strip():
        errors.append("run.json configurable_layers needs \"setting\": the environment variable or test parameter "
                      "that sets the layer count")

    if latest_entry_failure(run):
        warnings.append("run.json records a local failure of the entry test after its last pass: source statuses "
                        "decided from that pass rest on a superseded run; rerun the test or decide them again")
    elif run.get("entry_test_passes"):
        # A recorded pass covers every op the entry test launches (target-status.md).
        stale = [r["id"] for r in ops if r["origin"] == "static+measured" and r[f"{source}:status"] == "⚠️"
                 and r["launch_condition"].strip() == ALWAYS]
        if stale:
            warnings.append(f"op_table.csv {source}:status: rows {', '.join(stale)} ran in the measured entry test,"
                            " which run.json records as passing, but are ⚠️; decide them again"
                            " (merge_measured.py --entry-pass)")

    op_ids = {r["id"] for r in ops}
    for n, row in enumerate(rows["port_blockers.csv"], 1):
        if row["affected_ids"].strip() == "all":
            continue
        refs = [x.strip() for x in row["affected_ids"].split(",") if x.strip()]
        if not refs:
            errors.append(f"port_blockers.csv row {n}: affected_ids empty (comma-separated op_table ids or 'all')")
        for ref in refs:
            if ref not in op_ids:
                errors.append(f"port_blockers.csv row {n}: affected id '{ref}' is not an op_table id"
                              " (list ids separated by commas; ranges are not accepted)")
    for n, row in enumerate(rows["host_ops.csv"], 1):
        if row["counts_as_host_op"] not in HOST_OP_VALUES:
            errors.append(f"host_ops.csv row {n}: counts_as_host_op '{row['counts_as_host_op']}' not in {HOST_OP_VALUES}")
    if destinations:
        _check_device_op_support(run_dir / "device_op_support.csv", ops, destinations, errors, warnings)

    trace = rows["call_trace.csv"]
    for n, row in enumerate(trace, 1):
        if row["profile"] not in launch_targets:
            errors.append(f"call_trace.csv row {n}: profile '{row['profile']}' not in {launch_targets}")
    for target in launch_targets:
        op_total = conditional = measured = 0
        for r, count in zip(ops, launches[target]):
            if r["origin"] == "measured":
                measured += count or 0
                continue
            op_total += count or 0
            if r["launch_condition"].strip() != ALWAYS:
                conditional += count or 0
        trace_total = sum(
            (_int(r["repeats"], f"call_trace.csv row {r['id']} repeats", errors) or 0)
            * (_int(r["launches_per_repeat"], f"call_trace.csv row {r['id']} launches_per_repeat", errors) or 0)
            for r in trace if r["profile"] == target
        )
        if op_total != trace_total:
            errors.append(f"{target}: op_table launches {op_total} != call_trace launches {trace_total}")
        totals[target] = (op_total, conditional, measured)
    return errors, warnings, totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    errors, warnings, totals = validate(parser.parse_args().run_dir)
    for error in errors:
        print(f"ERROR: {error}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        return 1
    print("OK")
    for target, (total, conditional, measured) in totals.items():
        print(f"{target}: {total} launches per inference, {conditional} of them conditional "
              f"({total - conditional}-{total}); {measured} measured-only launches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
