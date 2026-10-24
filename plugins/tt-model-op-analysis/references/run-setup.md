# Run setup

## Interactive gate

Do not treat open files, IDE selections, shell history, environment variables or a previous run as
user confirmation.

Prefer the host's native structured-choice question tool when it is available (`AskUserQuestion`
in Claude Code or `request_user_input` in Codex when exposed). Use an actual tool call, not a
Markdown imitation of a form. Put the recommended option first, label it recommended and give one
short reason per option. If the tool supports only single choice, split a multi-select decision
into consecutive questions. If no such tool is exposed, show a short numbered list and say exactly
what reply is expected. Never invent an unavailable tool or stop merely because the rich UI is not
available.

Ask one decision per turn and stop for its answer. Skip values explicitly stated in the current
conversation. When discovery leaves exactly one compatible choice, state the choice and why it is
the only one, then continue without a redundant confirmation. Never ask the user to confirm the
whole configuration in one message.

### Setup order

1. Preconditions, before the first question: probe the local device (`probe_device.py` through
   `tt-device-mcp`), the elfray engine (measured-source.md, "Kernel cache") and TTsim readiness
   for Quasar (`ttsim.md`). None of them depends on the model. Measurement items without their
   hardware stay pending.
2. Ask for the source architecture: the hardware the source analysis runs on. Offer the profiles
   from `<plugin-root>/references/targets.md` with each one's one-line question; recommend the
   probed local device's profile, since only it can be measured here, and say which profiles have
   no local device (static analysis only) or need TTsim.
3. Ask which model to analyse. Do no model-specific discovery before the answer. Then ask for the
   output root (or state the configured one) and list the model's runs with
   `python3 <plugin-root>/scripts/runs.py list <root> <model>`: implementation, source profile,
   commit and date, static state, destinations and pending measurement items; mark the runs on the
   chosen source. Ask: continue one of them, or start a new run. A run in an older format (a
   `run.json` with `targets` or a single `destination`, plugin 0.3.0 or earlier) cannot be
   extended; say so and offer a new run. A run whose static state shows validator errors or
   warnings was written by an earlier plugin version: before continuing it, offer its migration
   first when it is feasible, then a new run ("Migrating a run" below).
4. New run, ref: ask which ref to pin. Recommend the local checkout's HEAD when a matching device
   or TTsim is available and the user is likely to measure (measurements must run on the run's
   commit, and a checkout at HEAD needs no switch or rebuild); otherwise recommend a freshly
   fetched `origin/main`. Say which and why. Resolve the ref before any discovery so every path,
   skip marker and CI claim comes from the pinned SHA.
5. Ask which implementation to analyse: every implementation directory of the model at the pinned
   SHA, in the table from section 1, each marked whether it fits the chosen source. Never hide a
   directory; recommend only one that fits. If the user picks one that does not fit, say why and
   ask again.
6. Rank the whole-model (end-to-end) test candidates of the chosen implementation directory for the
   chosen source. Recommend the best candidate with a concrete reason, then ask which test to use.
   Measurements rerun this same test.
7. If a run with this identity (model, implementation, entry test, source, commit) already exists,
   say so and continue it; never trace it again.
8. Write `run.json` with the run's identity (section 3) and `entry_test_ci`, the CI coverage that
   discovery found for the chosen test on the source profile. Leave out `kind`: the static phase
   sets it.
9. Entry test preflight (below), for a Blackhole or Wormhole source.

Destinations are asked after static analysis (static-phase.md, "Adding a destination"), never
during setup.

Do not inspect model source, create output or delegate tracing until the applicable setup is
complete. Candidate discovery may inspect directory structure, tests, skip markers and CI workflow
references at the pinned SHA, but not trace the model implementation yet.

## 1. Implementation and test

Ask which model. In the tt-metal checkout, find:

- every model directory (`git ls-tree -r --name-only <sha> models/ | grep -i <model>`),
- every implementation module (common, functional, interleaved, sharded, per-arch
  `blackhole/`, `wormhole/`, `t3000/`, `quasar/`),
- for each implementation, its whole-model test candidates (below), with the CI job that runs each
  one, if any (`.github/workflows/` and the pipeline lists in `tests/pipeline_reorg/*.yaml`, whose
  `skus` name the hardware),
- the skip markers on each of them: unconditional `skip` and every arch `skipif`
  (`is_blackhole`, `is_wormhole_b0`, ...), with their reasons.

A whole-model test runs the model forward from the model input through every block to the head
output. Confirm it by reading the test body: it calls the model's top-level forward (directly or
through a test-infra `run()`), not one layer, block or stage. Component tests
(`test_<model>_attention`, `_layer`, `_encoder`, `_embeddings`, ...) are never candidates.

Candidates: end-to-end PCC tests and demo tests, plus perf tests (device-perf: the
`command = f"pytest ..."` string in `*perf_device*.py` or `*device_perf*.py`; e2e perf such as
`test_*perf_e2e*.py`) whose test body runs the whole model, so one test serves static analysis and
measurements. For a Quasar source, candidates are whole-model tests of that implementation that run
on TTsim; no CI requirement. The entry test of a Quasar source is read, never run: not on a
device, not on TTsim.

An implementation is a directory of the model: one with its own `tt/` code, or a per-arch
directory (`blackhole/`, `wormhole/`, `t3000/`) whose tests and demo run another directory's `tt/`
code. List them all. Evaluate each, and later its entry test, against the chosen source profile.

Show one row per implementation directory: path, the `tt/` code its tests run, fit with the chosen
source (with the marker or skip that decides it), whole-model candidate node ids, skip markers and
whether CI runs them on the relevant architecture. Never derive the implementation
from a proposed test. Recommend an architecture-specific implementation exercised by CI or a demo,
not a reference or functional version. If several remain, prefer full target coverage and runnable
tests over naming or directory proximity.

After the user picks an implementation, rank its tests in this order:

1. runnable on the source target;
2. explicitly exercised by CI on the relevant architecture;
3. a warm-capable whole-model perf or PCC test, then a whole-model PCC/demo;
4. the least setup indirection and fewest unrelated parametrizations.

Present the top candidates with their node ids, relevant parameters, skip status, CI coverage and a
one-line recommendation reason. Never recommend a test skipped unconditionally or for a relevant architecture. If none is runnable,
say so and follow the temporary-test procedure instead of silently choosing another implementation.

### Setup traps (each misled an earlier setup)

- A directory's name is not its fit. ResNet-50's `quasar/` port branches on `is_blackhole()` and
  `is_blackhole_p100(device)` and runs on Blackhole; a `wormhole/` or `blackhole/` directory may
  only hold tests for shared code. Decide fit from the arch predicates in its `tt/` code and test
  infra and from its `run_for_*`, `skipif` and `pytest.skip` markers, and cite the one that decides.
- A defined skip helper is not a skip. Count a marker only where a test calls it.
- Offer choices from discovery at the pinned SHA, never from earlier runs: the model question is
  open, not a list of models that already have runs.
- A CI command can name a node id the test no longer generates (a device-perf `command` string
  with a dtype the test stopped parametrizing). Rebuild every proposed node id from the test's
  parametrization at the pinned SHA.
- CI is not only `.github/workflows/`: the pipeline lists in `tests/pipeline_reorg/*.yaml` run most
  model tests, and their `skus` say on which hardware.
- A native choice control holds at most four options. With more (five source profiles), offer
  the likeliest four and name the rest in the question for the free-text answer.

### No whole-model test

If the chosen implementation has no candidate of the required kind, say so and offer a temporary
test. Show its full content and stop for approval before writing it. It must:

- call the same top-level forward and weight preprocessing as the model's demo or component
  tests, with their batch size, input shape, dtypes and memory configs; say where each value
  comes from;
- use the existing `device` fixture and device params of the model's tests;
- carry no skip markers and no PCC threshold of its own (a PCC print is fine).

Where it lives:

- **static**: only in the run directory as `temp_test_<model>.py`, written once the output root
  is confirmed; never in the tt-metal checkout. Read it from there; every other source is still
  read as `git show <sha>:<path>`.
- **measured source phase**: copied as an untracked file next to the model's tests in the tt-metal
  checkout for the Tracy run, so its conftest and fixtures apply. Write it warm from the start (two
  forwards with `op-analysis-start` / `op-analysis-end` signposts). Delete it from the checkout
  after the run and say so.

Record it in `run.json` as `"test": "<run dir>/temp_test_<model>.py::<test name>"` and
`"temporary_test": true`, and state in `summary.md` that the entry test is temporary and not part
of tt-metal.

## Entry test preflight

The static phase is the expensive part of a run, and the entry test and the commit are part of the
run's identity. A test found to fail on the source device only after the static phase means a new
run and a second trace when the user then changes the test or the commit. The static analysis itself
stays valid either way: it reads code.

Offer the preflight only for a Blackhole or Wormhole source whose profile the local device probed
as. Never for a Quasar source: its entry test never runs, on a device or on TTsim. Without a
matching device, do not offer it; say that the test's state then rests on `entry_test_ci` alone and
continue with the static phase.

Ask one question: run the entry test once now, or skip. Recommend running it when `entry_test_ci`
is `no`, and skipping it when a CI job runs the test on this profile. With the question, state:

- the cost: the checkout must be at the pinned commit and built (`same_commit.py`; otherwise a
  checkout and a rebuild first), and the run compiles every kernel from nothing, up to
  `ci_pytest_timeout_s` or 1400 s, with the measured-source phase's 900 s device-operation
  watchdog for the cold capture;
- the risk of skipping: the static phase goes ahead on a test nobody ran here; if it later fails on
  this device, the measurements have nothing to read, and moving to another test or commit is a
  new run, traced again.

The preflight is the capture run of measured-source.md (steps 1 to 4, `source:graph` and
`source:cache` in one run; the cache is built even without the elfray engine and read later), so it
costs no extra run. One job through `tt-device-mcp`. Then:

- **Passed.** `python3 <plugin-root>/scripts/graph_report.py <capture file> <run dir> --inferences <n> --date <YYYY-MM-DD> --entry-pass "<pass line>"`
  records the pass in `entry_test_passes` and the capture in `graph_capture`; with no
  `op_table.csv` yet it decides no row. Run `cache_report.py` (measured-source.md, step 9) when the
  engine is available. Continue with the static phase.
- **Failed.** Show the failure line and ask: another entry test (step 6), another commit (step 4),
  or the static phase anyway. For the last, record the failure with `graph_report.py ... --entry-fail "<failure line>"`
  when the capture file was written, otherwise with
  `python3 <plugin-root>/scripts/measurements.py mark <run dir> source:graph failed --note "<failure line>"`.
  For another test or commit, delete this `run.json` and the capture and cache with the user's
  approval, and set the run up again.

When a model has a Blackhole and a Quasar implementation (for example
`models/demos/vision/classification/resnet50/ttnn_resnet` and `.../resnet50/quasar`), each is its
own run with its own source. Comparing the two is a separate step on two runs.

## 2. Pin the source

Record the full SHA of the commit analysed (`git rev-parse <ref>`; the recommendation is in the
setup order, step 4). Read every file as `git show <sha>:<path>`; never read
the working tree, except an approved temporary test (section 1). For Quasar, record the ttsim
version if the user knows it.

zsh trap: `$VAR:path` applies the `:t`/`:h` modifiers. Write `<sha>:path` with the literal SHA or
use `"${VAR}:path"`.

Treat `main`, `origin/main` and `HEAD` as different refs; never silently substitute one for another.
When the user says `main`, resolve local `main` exactly, and report whether an already-present
`origin/main` points elsewhere and which is newer by ancestry. Fetch only when the user selected a
fresh remote ref and the environment permits it. Record both the symbolic ref and resolved SHA in
the run notes so a later comparison can detect a stale or divergent pin.

## 3. Output location and run identity

Root: current working directory unless the user names one; never inside the tt-metal checkout.

- Run: `<root>/op-analysis/<model>/<impl-slug>/<source>/<sha7>/`. `impl-slug` is the
  implementation module's directory name, or its parent's name when that directory is `tt`
  (`ttnn_resnet`, `quasar`). One directory holds the source trace, every destination and every
  measurement of that run.
- Identity: model, implementation module, entry test, source profile and `tt_metal_commit`. The
  same model, implementation and source on another commit is a new run. There is no overwrite or
  merge: a run with the same identity is continued.

`run.json` fields:

```json
{"kind": "static", "model": "resnet50", "implementation": "<module path>", "test": "<node id>",
 "source": "p150", "destinations": ["quasar-bf16"], "stage_blocks": ["stem", "layer1", "head"],
 "profile_values": {}, "tt_metal_ref": "origin/main", "tt_metal_commit": "<40-char sha>",
 "date": "YYYY-MM-DD", "drive_url": null, "entry_test_passes": {},
 "entry_test_ci": {"value": "yes", "evidence": "tests/pipeline_reorg/models.yaml: resnet50 (skus p150)"}}
```

`entry_test_ci`, written at setup: whether a CI workflow or pipeline list runs the entry test on the
source profile at the pinned commit. `value` is `yes` or `no`; `evidence` names the workflow or
pipeline file and job for `yes`, and for `no` what discovery found (no job, or jobs on other
profiles only, named). For a Quasar source `n/a`, with the evidence that its entry test is read,
never run.

Static analysis adds `configurable_grid` and `configurable_layers` (static-phase.md, step 6).
`entry_test_failure` (`{"by", "date", "result"}`) records a failed local run of the entry test
(`--entry-fail` of `merge_measured.py` or `graph_report.py`); a later recorded pass removes it.

`entry_test_passes`, for a Blackhole or Wormhole source: where the entry test is shown to pass,
either `{"ci": "<workflow file>: <job>"}` or `{"by": "...", "date": "...", "result": "<pcc line>"}`
for a run the user reports or a passing Tracy run (`merge_measured.py --entry-pass` writes
`"by": "source:tracy on a local <source> device"` and keeps a `ci` entry). It backs the `✅` rule in
`target-status.md`.

Measurements add: `measured_ops` (path of `measured_ops.csv`, set by `merge_measured.py`),
`measured_inferences`, `tracy_mode` (`live` or `trace`), `ops_csv` (the raw Tracy report), `window` (`warm` or `cold`),
`ci_pytest_timeout_s` (the CI `pytest_timeout` of the entry test, when it has one), `graph_capture`
(set by `graph_report.py`: capture path, date, inferences, launch and kernel-op counts, host
transfers inside the forward, capturing commit), `kernel_cache` (set by `cache_report.py`: cache
path, date, elfray engine, layers, build and row counts), `measurements`
(marks from `measurements.py mark`), and for TTsim `ttsim_checkout`, `ttsim_commit`,
`simulator_library`, `simulator_arch` and `ttsim_environment` (the exact non-secret variables
`kernel_tests.py run` sets).

Other runs are never evidence: during inventory, tracing and status decisions, read no file of
another run under `op-analysis/`. The run being continued is different:
the run being continued is the input for adding a destination or a measurement, and its tables are
read and extended. The scripts (`runs.py`, `compare_runs.py`) may open any run.

On request, two runs can be compared table by table:
`python3 <plugin-root>/scripts/compare_runs.py --old <old>/<t>.csv --new <new>/<t>.csv --out <new>/changes_<t>.csv --key <key>`.

| Table | `--key` |
|---|---|
| `op_table.csv` | default (`stage,ttnn_api,device_op,program_factory`) |
| `call_trace.csv` | `profile,stage` |
| `port_blockers.csv` | `blocker` |
| `host_ops.csv` | `host_work` |
| `host_fallback.csv` | `scope` |
| `footprint.csv` | `op_code` |
| `kernel_ops.csv` | `device_op,program_factory,op_code` |
| `device_ops.csv` | `device_op` |
| `kernel_test_results_<target>.csv` | `node_id` |

`measured_ops.csv` is not compared: per-op timings change every run. `compare_runs.py` exits 2 if a
key column is missing.

### Interpreting a cross-run comparison

`compare_runs.py` is a syntactic table diff, not a semantic verdict. Before describing a change:

1. Check commit ancestry and dates. State when the nominally new run pins an older, stale or
   divergent ref.
2. Compare blob IDs for the model and entry test, then inspect whether the dispatch/factory sources
   cited by changed rows differ between SHAs.
3. Normalize representation changes: zero-launch rows versus notes, split versus merged identical
   variants, stage naming, `prim::` versus `ttnn::prim::`, and explanatory text embedded in key
   columns. Do not report these as model changes.
4. Compare launch totals and execution order before support-status counts. Reconcile status counts
   on the same set of unique launching variants.
5. For each remaining status/confidence disagreement, reopen pinned evidence from both commits and
   decide whether it is a source change or an analysis-quality difference. State which result is
   better supported; do not average the two reports.

Never use an earlier report as dispatch evidence for a new run. It is only a list of claims to
re-check against pinned source.

## 4. Migrating a run

A run written by an earlier plugin version keeps its identity (model, implementation, entry test,
source, commit) but can miss what the current version records or break a rule added since. The
inventory shows it as validator errors or warnings in its static state. Offer, in this order:

1. **Migrate** (recommended) when every fix can come from the run's own files and the tt-metal
   source at its pinned commit: no new trace of the model and no new device run.
2. **New run** when a fix needs what the run cannot give (a trace of a call the run never followed,
   a measurement whose input is gone), or the run is in the 0.3.0-or-earlier format.

Say which fixes each choice involves before asking.

To migrate:

1. Run `python3 <plugin-root>/scripts/validate_static.py <run dir>` and fix every error and warning.
   Reopen the source as `git show <sha>:<path>` at the run's `tt_metal_commit` for any fix that is
   a decision, never the working tree or another run:
   - `grid_dependency` with text after `automatic` or `none`: decide per row from the pinned source
     whether ttnn derives the grid from the device (the bare keyword) or the model sets it (the
     value with its `<file>:<line>`). A row that cites a model file is not derived until the
     source shows it.
   - Missing `configurable_grid` or `configurable_layers`: static-phase.md, step 6.
   - Missing `tracy_mode`: `live` when the entry test calls the forward directly, `trace` when the
     measured window replays a captured device trace.
2. Regenerate what newer scripts derive from inputs the run already has: `kernel_ops.py`;
   `diff_reports.py` when `measured_ops.csv` exists; `graph_report.py` on the recorded capture file
   when it still exists (peak DRAM). An input that is gone stays pending; do not rerun the device.
3. Write every change with the scripts or a heredoc; never retype a table. Rerun the validator
   until it prints `OK`, then update `summary.md` where its counts or statements changed, and list
   what the migration changed and decided. Uploading the result is the `upload-op-analysis` skill.
