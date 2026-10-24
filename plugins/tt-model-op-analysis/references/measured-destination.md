# Measured destination phase

Runs a Quasar target's kernel-op tests on TTsim and replaces assumed results with run results. The
target is a Quasar destination (`quasar`, `quasar-bf16`) or a Quasar source; pass it as
`--target <target>` (it may be omitted when the run has only one Quasar target). Each target has
its own `kernel_tests_plan_<target>.csv` and `kernel_test_results_<target>.csv`.
Do not run the whole-model test on a destination; this phase runs kernel-op tests one by one, each with
the parameters written in the test. A test decides a row's status when it runs the same kernel op (device
op, program factory, layout and sharding, dtype; target-status.md). Its shapes, batch and core grid may
differ from the row's and no script compares them, so the evidence names the differences. A
Quasar source's entry test is its own item (measured-source.md).

1. Readiness: `python3 <plugin-root>/scripts/same_commit.py <run dir> --tt-metal <checkout>` must
   pass (`kernel_tests.py run` checks it again). Follow `<plugin-root>/references/ttsim.md`
   discovery and readiness for Quasar, and record `ttsim_environment` (the exact non-secret
   variables) in `run.json`.
2. Kernel-op tests: every row's `<target>:tests` lists the tests that decide its status. For each
   `⚠️` row with no test, search the Quasar tests (the model's Quasar op tests,
   `tests/ttnn/**/quasar/`, `tests/scripts/quasar/*.yaml`) for one that calls the same kernel op
   (device op, program factory, layout, dtype); add its node id and say so in the evidence. Do not
   write new tests.
3. `python3 <plugin-root>/scripts/kernel_tests.py plan <run dir> --target <target>` lists the unique
   node ids with the statuses of the rows that cite them. Then
   `python3 <plugin-root>/scripts/kernel_tests.py check <run dir> --target <target> --tt-metal <checkout> --pytest-cmd "<interpreter> -m pytest" --fix-node-ids`
   (with `--env-json <file>` for a batch with its own environment) collects each test file of the
   plan on the host. The fix flag rewrites a parametrized node id only when exactly one collected
   id for the same function differs by an added prefix, such as `quasar-`, then regenerates the
   plan. It lists every remaining mismatch with the collected ids of the same test function.
   Correct ambiguous mismatches in `op_table.csv` `<target>:tests`, or in the script that wrote
   that column, and rerun `plan` and `check`; `run` refuses an uncollected node (exit code 3). Then
   `python3 <plugin-root>/scripts/measurements.py list <run dir> --pending` shows the items not
   yet done (`<target>:kernel-tests ❌`, `<target>:kernel-tests ⚠️`, `<target>:kernel-tests ✅`)
   with node counts and an estimate; a node whose result is `error` or `timeout` counts as not
   done. Until one test has run the estimate is unknown: offer to run one test first
   (`--only <text>`) so the rest is estimated from its duration. Stop for the user's pick.
4. `python3 <plugin-root>/scripts/kernel_tests.py run <run dir> --target <target> --tt-metal <checkout> --pytest-cmd "<interpreter> -m pytest"`
   runs each node with plain pytest on the host under `ttsim_environment`, one process per node,
   never through `tt-device-mcp`. It sets no wall-clock limit and disables pytest-timeout
   (ttsim.md, "Run time"): run it in the background and let it finish. `--timeout <s>` is for a
   user who accepts losing the result; a node it kills is recorded as `timeout`, which decides
   nothing. `--status ❌` or `--status ⚠️` runs one item's nodes. Without `--pytest-cmd` it uses
   `<checkout>/python_env/bin/python` when present; the command used is recorded in `run.json`.
   It resumes after an interruption: nodes with an `error` or `timeout` run again, the others only
   with `--rerun`.
   Tests that need different variables run as separate batches: `--only <text>` selects plan
   nodes, `--env-json <file>` gives that batch's environment, and earlier batches' results are
   kept.
5. Read the logs of every `error` result before applying: an error means the test could not run
   (environment, import), not that the kernel op fails. Fix the environment and rerun those nodes.
   Report a `timeout` as unresolved with its duration and the last log line, never as a hang.
6. `python3 <plugin-root>/scripts/kernel_tests.py apply <run dir> --target <target>` sets
   `<target>:test_result` and the status per row: a `fail` or `xfail` of any of its tests gives
   `❌`; `✅` needs every test of the row to pass; `timeout`, `error`, `skip` and unrun tests leave
   the status unchanged. The evidence starts with `TTsim run: <node> <result>`.
   `apply` overwrites the status and keeps the static evidence after ` || `, but not the static
   status. To withdraw a result: delete its row from the results file; in `op_table.csv`, set each
   row it decided back to its status per target-status.md and drop the `TTsim run: ... || ` prefix
   from its evidence (or regenerate the table with the script that wrote it); then run `apply`
   again. `apply` warns about every row whose status still rests on a TTsim result the results
   file no longer decides. A results file from plugin 0.5.0 may hold `hang`: it is read as
   `timeout`, so the node runs again and the row is withdrawn as above.
7. A TTsim pass of any test of a device op makes its `device_op_support.csv` verdict `✅`
   (target-status.md, "Device op support"); update it. Rerun `validate_static.py` and
   `kernel_ops.py`. In `summary.md`, state which statuses rest on a TTsim run and which on test
   markers only.

TTsim proves execution and correctness on the simulated architecture, never performance.
