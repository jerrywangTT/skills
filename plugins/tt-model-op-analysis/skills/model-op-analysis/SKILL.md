---
name: model-op-analysis
description: Analyse the kernel ops one implementation of a tt-metal model runs on its architecture (P100, P150, N150, one chip of an N300, or Quasar) and how they would port to other architectures. Traces every ttnn call to its device op and program factory at a pinned commit, decides support per kernel op, optionally adds destinations (same code on another profile, or a Quasar BF16 projection) without re-tracing, and resumes measurements item by item - runs of the entry test on the source device that yield a Tracy profile, a ttnn graph capture (every launch's device op and program factory) and a kernel cache decomposed with elfray (LLK and dataflow APIs per kernel), and TTsim runs of Quasar kernel-op tests. Use for "op table", "which kernel ops does model X use", "can model X run on Quasar", or kernel-op support and porting analysis.
---

# Model op analysis

One run = one implementation of a model, its entry test, a source profile and a tt-metal commit,
in `<root>/op-analysis/<model>/<impl-slug>/<source>/<sha7>/`. Static analysis is done once per
run; destinations and measurements are added to it, in this session or a later one.

`<plugin-root>` is the `tt-model-op-analysis` plugin directory, two levels above this SKILL.md;
resolve every `<plugin-root>/...` path against it.

## Flow

1. **Preconditions, first.** Before any question, probe what measurements could use: the local
   device (through `tt-device-mcp`), the elfray engine for the kernel cache
   (`<plugin-root>/references/measured-source.md`, "Kernel cache") and TTsim readiness for Quasar
   (`<plugin-root>/references/ttsim.md`). Report what is available.
2. **Source architecture.** Ask which hardware the source analysis runs on, recommending the probed
   local device's profile (`<plugin-root>/references/run-setup.md`, "Setup order").
3. **Model and inventory.** Ask for the model, then for the output root (state it instead when the
   user's instructions fix it), then run `python3 <plugin-root>/scripts/runs.py list <root> <model>` and show its runs: implementation,
   source, commit, static state, destinations, pending measurements. Ask: continue a run, or start
   a new one. A run in an older format (plugin 0.3.0 or earlier) cannot be extended; offer a new
   run. A run whose static state shows validator errors or warnings comes from an earlier plugin
   version: offer its migration first when feasible, then a new run
   (`<plugin-root>/references/run-setup.md`, "Migrating a run").
4. **New run.** Follow `<plugin-root>/references/run-setup.md`: ref, then the implementation
   (every implementation directory of the model, each marked whether it fits the source), then an
   end-to-end entry test from that directory. If the same identity exists, say so and continue that
   run instead of tracing again. Then write `run.json` with the identity and the test's CI coverage
   (`entry_test_ci`).
5. **Entry test preflight.** For a Blackhole or Wormhole source with a matching local device, ask
   whether to run the entry test once before the static phase, with its cost and what skipping
   risks (`<plugin-root>/references/run-setup.md`, "Entry test preflight"). The run is the
   graph-and-cache capture run of the measured phase, so it is not repeated later. Never for a
   Quasar source: an entry test never runs on TTsim.
6. **Static.** Follow `<plugin-root>/references/static-phase.md`; it ends when
   `python3 <plugin-root>/scripts/validate_static.py <run dir>` prints `OK`. Skipped for a run whose
   static analysis is done.
7. **Destinations.** Ask whether to add a destination; repeat while the user wants more. Offer every
   choice but the source in one list: `p100`, `p150`, `n150`, `n300-1chip`, `quasar` (the code as
   written, on Quasar) and, for a non-Quasar source, `quasar-bf16` (BF16 projection). Each one is
   added to the same run from the existing source trace (static-phase.md, "Adding a destination").
8. **Measurements.** First the source methods: probe the source device again, then run
   `python3 <plugin-root>/scripts/source_methods.py <run dir> --device <probed profile or none> --tt-metal <checkout>`
   and show all four analysis methods (static trace, Tracy, graph capture, kernel cache) with
   their state: done, available, available to read later, read an existing cache, or unavailable
   and why. Let the user pick
   any number of the available ones in one multi-select question, all of them recommended. The
   source items `source:tracy`, `source:graph` and `source:cache` come from two runs of the entry
   test (graph and cache together, Tracy alone). For an unavailable item offer to leave it pending
   or record it with
   `python3 <plugin-root>/scripts/measurements.py mark <run dir> <item> failed --cause <test|tool|not-run> --note "<reason>"`
   (measured-source.md says which cause).
   Then run `python3 <plugin-root>/scripts/measurements.py list <run dir> --pending` and offer the
   destination items the same way, never ones already done.
   Source items follow `<plugin-root>/references/measured-source.md`
   (`python3 <plugin-root>/scripts/same_commit.py` first, then the device runs, then
   `<plugin-root>/scripts/merge_measured.py`, `<plugin-root>/scripts/graph_report.py`,
   `<plugin-root>/scripts/cache_report.py` and the Quasar LLK gap check
   `<plugin-root>/scripts/llk_gaps.py` for the items picked); Quasar test items follow
   `<plugin-root>/references/measured-destination.md` (`python3 <plugin-root>/scripts/kernel_tests.py`).
   Unpicked items stay pending for a later session.
9. **Report.** `python3 <plugin-root>/scripts/kernel_ops.py <run dir>`, the validator again, and
   `summary.md` with the per-kernel-op table first. `summary.md` names the methods the run used and
   the confidence they give (`<plugin-root>/references/schemas.md`, "Analysis methods and
   confidence"). Say where the results are; uploading them to Google Drive is the
   `upload-op-analysis` skill, which the user starts when they want it.

## Rules

- Before ending a session that changed the run, append your entry to `run.json` `analysed_by`
  (agent, model, the work done, date; run-setup.md, section 3). Never edit another agent's entry.
- Ask one decision per turn, recommended option first; never infer choices from open files, IDE
  state or earlier runs.
- Read tt-metal sources as `git show <sha>:<path>`; never the working tree, except an approved
  temporary test (run-setup.md).
- Measurements run on the run's `tt_metal_commit` only; `same_commit.py` is the gate.
- Physical-device commands go through `tt-device-mcp`; TTsim runs on the host with Bash, never
  through `tt-device-mcp`, even under a rule that routes all pytest there (`ttsim.md`).
- TTsim runs only kernel-op tests, one node at a time; never an entry test or any whole-model test.
- Device runs are sequential: one device job at a time (probe, capture run, Tracy run), each started
  only after the previous one has finished and the device reports healthy. Never submit a second
  device job while one runs or is queued, and never give device runs to parallel subagents: two
  workloads on one device can hang it and leave it held. TTsim tests also run one at a time.
- Subagents follow the contract in `<plugin-root>/references/dispatch-tracing.md`: no nested
  subagents, one complete result.
- Write CSVs with the scripts or a heredoc into the run directory; never retype a table to fix it.
- Do not edit the tt-metal checkout, except an approved temporary test (run-setup.md).
- Output stands alone: a reader who knows no issue, ticket, tracker or earlier conversation must
  understand every file. Never refer to criteria, items or numbers defined elsewhere; state what
  is meant. A link may cite a source, never replace a definition.
- Report what the validator checked, what stayed `unverified`, which statuses rest on a TTsim
  run versus test markers, and which kernel ops a graph capture or Tracy run saw that the static
  trace did not.
