# Target status rules

Every target of a run (`targets.md`) has two columns in `op_table.csv`: `<t>:status` and
`<t>:evidence`. Evidence belongs only to its own target's status; never share one evidence column
between targets.

A status answers one question: is this op supported on this target? It does not answer whether
the launch happens or which factory is chosen; `launch_condition` and `confidence` carry that.
Do not downgrade a status because dispatch is `unverified` or the launch is conditional.

| Value | Meaning | `<t>:evidence` must name |
|---|---|---|
| `✅` | Supported, shown by a test or recorded run a reader can open (rules per target below) | the test or run and why it matches this row |
| `⚠️` | The code path exists, but no test or run covers this variant on this target | the factory or fork path, and what coverage is missing |
| `❌` | Directly proven not to work: API/binding mismatch, validation failure, TT_FATAL, or an exact reproducer of a hang or wrong result | the failing check or reproducer |
| `n/a` | The target never executes this op (untaken branch; for a target with launch counts exactly when launches is 0) | why it is not executed |

Apply the definitions as written. Absence of a test is never evidence of failure. For the
Quasar targets the definitions are made concrete per kernel op in "Quasar status by kernel op"
below.

## p100, p150, n150, n300-1chip

The code runs as written on that profile, as source or as a destination.

- `✅`: the entry test passes on this profile, either in a CI workflow that runs it on this
  profile (name the workflow file and job) or in a passing run recorded in `run.json`
  `entry_test_passes`. Every op the test launches is then covered. A dedicated op test on this
  profile with the same variant also counts. A passing `source:tracy` run is such a recorded run;
  when no CI workflow runs the test on this profile, the evidence says so and names the local run
  ("No CI workflow runs the entry test on P100; it passed on a local P100 under Tracy (<pass
  line>, <date>) ..."). It replaces a `⚠️` that only said no passing run was found
  (measured-source.md, step 5).
- `⚠️`: the profile is reached only through a branch the entry test does not run on this profile,
  or no passing run exists for this profile.
- Branch-dependent rows (a reshard that only P150 needs) are `n/a` with 0 launches on the other
  profile.
- For a destination profile, `entry_test_passes` names where the entry test passes on that
  profile; without one, rows are `⚠️` unless an op test with the same variant runs there.

## quasar as the source

The analysed implementation is Quasar code (for example a model's Quasar port). Status follows
"Quasar status by kernel op" for each row's own kernel op. The entry test never runs on TTsim;
one whole-model pass or failure would not say which kernel op caused it.

## quasar as a destination (code unchanged)

The analysed code unchanged on Quasar: same dtypes, grids and mainline ttnn calls. Mainline calls
are not routed to Quasar forks automatically. Each row keeps the kernel op and parameters the source traced: a branch the
code takes on Quasar (`is_quasar()`, an arch check in a fork) neither drops the row nor makes it `n/a`. Most rows are usually `❌` for a reason shared by
many rows; state the specific check once in `port_blockers.csv` and cite its id plus the row's own
detail (dtype, factory family) in the evidence.

## quasar-bf16 (projection)

Not existing code. The status the op would have if the analysed code changed in exactly two ways:
every block-float dtype (`bfloat8_b`, `bfloat4_b`) becomes `bfloat16`, and every mainline call
that has a `ttnn.experimental.quasar.*` fork calls that fork instead. A mechanical substitution to
the fork's corresponding program-config type is part of that routing only when its fields map
one-to-one; all values and topology stay unchanged. No retuning, algorithm change or missing
argument may be assumed. Grids, shard specs and program-config values stay as written; grid
blockers go to `port_blockers.csv`. The evidence names the dtype change, fork and config-type
mapping, or says none applies. `✅` means a Quasar test covers the projected kernel op, not that the
projected model runs.

If a fork lacks a required argument or has no one-to-one program-config type, the projection is
`❌`; the signature mismatch is direct evidence. Do not silently redesign the model.

## Quasar status by kernel op

Every Quasar target (`quasar` as source or destination, `quasar-bf16`) decides status per kernel
op. The kernel op of a row is its **device op + program factory + layout/sharding + dtype** as it
would run on Quasar (for `quasar-bf16` after the bf16 change and fork routing). Batch size, core grid and shard sizes are not part of the key; record them in
the evidence as the test's scope. They become part of the key only when a documented failure
depends on them (for example a block-sharded matmul that fails only when K is split across
several cores, `in0_num_blocks_w > 1`).

1. Find the Quasar tests that call this kernel op: the model's Quasar op tests, the Quasar unit
   tests under `tests/ttnn/**/quasar/`, and the rows of `tests/scripts/quasar/*.yaml`. Read each
   candidate test: its parameters, markers and comments.
2. Read the release documents at the pinned SHA (`releases/<version>/*milestones-and-tests*.md`)
   for the tests found and for any end-to-end model run on Quasar. They record what was run on a
   simulated Quasar for a release, so they outrank test comments (see "Sources of evidence").
3. Decide:

| Status | When | Evidence names |
|---|---|---|
| `✅` | A test runs this kernel op (same op, factory, layout, dtype) with no `xfail`/`skip` marker, is not a hang reproducer, and has no comment reporting that it fails | the test node and parameters; the batch/grid it uses when they differ from the row |
| `❌` | A test or reproducer for this kernel op is marked `xfail`, is a hang reproducer, or carries a comment reporting a hang, wrong result or error; or the Quasar code has no factory for this combination (validate TT_FATAL, no fork and only a ProgramDescriptor/legacy factory, unsupported dtype) | the marker, reproducer, comment or failing check with `file:line` |
| `⚠️` | No test runs this kernel op, but the Quasar fork has a Metal 2.0 ProgramSpec factory whose validate accepts this layout and dtype | the factory and the scope its validate supports (layouts, dtypes, sharding), and that no test exists |
| `n/a` | The target never executes this op | why |

- A test without a marker is assumed to pass; results of `tests/scripts/quasar/*.yaml` are only in
  an external emulator check. Say whether the test is in CI, in a yaml list, or in neither.
- `<t>:tests` holds the node ids of the tests that decide the status; `<t>:test_result` stays
  `not run` until `kernel_tests.py` runs them on TTsim (measured-destination.md). A run result
  overrides the marker rule: a `fail` or `xfail` of any of the row's tests gives `❌`; `✅` needs
  every test of the row to pass. `timeout` (killed by a `--timeout` the user chose; on TTsim a
  slow run and a hang look the same), `error` (the test could not run, for example a broken
  environment), `skip` and unrun tests decide nothing: the status stays as decided here.
- "Hang reproducer" and "reproduces a hang" in this file mean a test whose marker or comment
  documents a hang. A TTsim run that was slow or killed never makes a test one.
- A comment or docstring that blames a missing or unfinished port ("fatals until X is ported")
  counts only while the pinned code still lacks X. Look for X at the pinned SHA; when it exists,
  the comment is stale: say so in the evidence and decide from the code and markers instead.
- When tests disagree (one passes, another reproduces a hang for the same kernel op), use `❌` and
  cite both; a comment elsewhere saying "fixed" does not override a live reproducer without a run.
- A failure applies to every row with the same kernel op key. A failure for another factory,
  layout or dtype does not.

## Device op support (destinations)

Row statuses judge the model's own calls. When none of a device op's rows is `✅` on a destination,
that alone cannot say whether the destination lacks the op or only these arguments: the model's
calls are a sample. For every destination, decide per device op (`op_table.csv` `device_op`, or
`op_code` when it is empty) whether the destination runs the op at all, for any arguments, and
write it to `device_op_support.csv`:

| `<t>:op_support` | When | `<t>:op_evidence` names |
|---|---|---|
| `✅` | A test on this destination runs this device op with any parameters and passes, or has no marker, hang reproducer or failure comment; or a row of the op is `✅` | the test node and how its parameters differ from the model's calls |
| `⚠️` | No test of any variant runs the op on this destination, but the destination has code for it (a factory, for Quasar a Metal 2.0 ProgramSpec factory or fork, whose validate accepts some input) | the factory and that no test was found, with where was searched |
| `❌` | A check fails whatever the arguments: no binding, no factory or fork for this architecture, every factory of the op stops on it (legacy `CreateKernel` factories on Quasar) | the check with `<file>:<line>`, or the `port_blockers.csv` id that states it |
| `n/a` | Every row of the op is `n/a` on this destination | why |

- Search the whole repository at the pinned SHA, not only tests matched to the model's rows: for
  Quasar `tests/ttnn/**/quasar/`, `tests/scripts/quasar/*.yaml` and the release documents; for a
  profile, the op's unit tests and the CI workflows that run them on that profile. Say what was
  searched when nothing was found ("no test under tests/ttnn/**/quasar/ or tests/scripts/quasar/
  calls ttnn.experimental.quasar.conv2d").
- A failure tied to arguments (a dtype, a layout, `in0_num_blocks_w > 1`, an `xfail` on one
  parametrization) is a row status, never `❌` here. A check whose condition reads any tensor
  attribute or op argument is tied to arguments.
- Absence of a test is `⚠️`, never `❌`.
- A device op with several factories is `✅` when any factory runs on the destination; the
  evidence names the factories that do not.
- A TTsim pass of any test of the op gives `✅`; update the verdict after `kernel_tests.py apply`.
  When that result is withdrawn, decide the verdict again from the remaining evidence.
- Leave a verdict empty only while it is not decided; the sheet shows it as `⏳ not checked` and
  the validator warns.
- `validate_static.py` rejects `⚠️` or `❌` for an op with a `✅` row, `❌` unless every launched
  row is `❌` (a failure for any arguments fails every call), and `n/a` unless every row is `n/a`.

## Sources of evidence

Strongest first; a stronger source decides over a weaker one for the same kernel op key:

1. A TTsim run of this analysis (`kernel_tests.py`); the evidence starts with `TTsim run:`.
2. A release document at the pinned SHA. A test it lists as passing gives `✅`; one it lists as not
   supported gives `❌`, unless a later milestone of the same or a later document verifies it. An
   end-to-end run it records as verified gives `✅` to every kernel op key that run launches,
   traced through the model it ran: same op, factory, layout and dtype, whatever its batch or grid.
   When that run needed an environment workaround on this path (a forced program route, a debug
   mask over a known hazard), the status is `⚠️` and the evidence names the workaround. The
   evidence starts with `Release <version> <milestone>:`.
3. Test markers and comments, as in the table above.
4. The fork code alone (`⚠️` for a usable factory, `❌` for a missing one).

A release record that contradicts a marker at the pinned SHA (a test listed as passing but marked
`xfail`) gives `⚠️` citing both. `summary.md` says how many `✅` rest on each source.

## Quasar facts to verify at the pinned commit

- Block-float formats: `is_supported_quasar` in `tt_metal/common/tt_backend_api_types.cpp` lists
  the formats Quasar accepts; at the time of writing it has no `Bfp8_b`/`Bfp4_b`.
- Legacy kernels: `DataMovementKernel`/`ComputeKernel` constructors in
  `tt_metal/impl/kernels/kernel.hpp` TT_FATAL on Quasar, so ProgramDescriptor/CreateKernel
  factories cannot run there; only Metal 2.0 ProgramSpec factories lower to Quasar kernels.
- Quasar forks live under `ttnn/cpp/ttnn/operations/experimental/quasar/` and are exposed as
  `ttnn.experimental.quasar.*`.
- Quasar op tests are mostly outside CI (`tests/scripts/quasar/*.yaml`); say so when a `✅` rests
  on a locally run test.

## Coverage matching and consistency

Match the semantics that affect dispatch: required arguments and output behavior, in-place versus
out-of-place, broadcast mode, sharding/layout, dtype, fused bias/activation, and factory family
(for example 1D versus 2D multicast). A test of fold without `collapse_output` does not cover a
call that requires `collapse_output`; a 1D-mcast linear test does not cover a 2D-mcast linear.
Memory placement alone may differ.

Before finishing, group rows by the target-relevant signature (operation semantics, factory,
layout, dtype and sharding). Equal signatures must have equal statuses unless the evidence states
the row-specific distinction. Rows with one signature share one evidence text, written once and
copied: the decisive fact and its `<file>:<line>` in one sentence of at most 300 characters, never
prefixed with the row number or stage (the sheet shows both). Put what distinguishes a row in its
own text. The validator warns about `Row <n>` prefixes and longer text.
