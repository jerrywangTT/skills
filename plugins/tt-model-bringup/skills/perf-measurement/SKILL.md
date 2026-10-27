---
name: perf-measurement
description: "Measure TTNN performance changes so that a win is real and attributable: back-to-back A/B on one build with a noise rule, one code path per measurement, per-chip timing on meshes, real routing data for mixture-of-experts timing, pinned prompts and auto matmul configs, output checks, and real-time-profiler perf tests for op changes. Use before accepting any speedup, when numbers disagree between runs or machines, or when an op change needs a perf test."
---

# Performance measurement

Before using this workflow, follow the dependency and environment setup in
[the model-bringup entrypoint](../model-bringup/SKILL.md#startup).
Resolve its package root from this installed skill's location; do not assume
this plugin is copied into the target tt-metal checkout.

Each check below changed a conclusion while optimizing gpt-oss-20b on a QuietBox 2 (4 Blackhole chips;
tenstorrent/tt-metal #60161, #61050, #61548, #62558–#62566). Account against the roofline as in
[Performance Accounting](../optimize/SKILL.md#performance-accounting).

## PM-01: Compare A and B back to back, and beat the noise

**Do:** run before and after on the same build, board and harness, alternating, at least twice each. Accept a change
only if the A–B difference exceeds the spread between repeats of the same code; report the repeats with any win under
about 2%. Prefer a temporary switch inside one build to two builds, and remove it before committing.

**Why:** identical code varied about 1% between rounds (414.5 vs 418.1 tok/s/user), as much as many single changes.

## PM-02: Measure one code path at a time

**Do:** time each path over the full range of inputs it will serve. Do not hide a threshold that silently falls back to
another path inside the thing being measured.

**Why:** with a fallback, a sweep reports the other path's numbers for inputs the new path never ran.

## PM-03: Time every chip on a mesh

**Do:** compare the same op across chips: the `DEVICE ID` column of the device-profiler ops report, or the `chip_id` of
real-time profiler records.

**Why:** collectives make every chip wait for the slowest, and a model-level or one-chip view hides it. Two of four chips
1.13–1.34x slower on every weight-streaming op cost 20% of batch-1 decode ($decode-weight-streaming WS-04).

## PM-04: Time mixture-of-experts layers on real text

**Do:** use real prompts and real K/V contents. If full prefills are too slow, prefill a few real documents once and
point users' page tables at them.

**Why:** filler tokens (repeated K/V) send most tokens to the same few experts and made batched decode up to 25% too fast.

## PM-05: Pin what changes between runs

**Do:** pin the date if the chat template stamps it (gpt-oss does), or average over several dates. For `ttnn.matmul`
calls with auto-selected configs, run the call once with L1 free, take the program config it chose, and pass that
config explicitly.

**Why:** the date alone changed speculative-decoding acceptance between days. The auto config sizes its blocks to the
L1 free at the call, so the same matmul ran a different program (with different rounding and speed) depending on what
else was in L1.

## PM-06: Check outputs, not only speed

**Do:** for changes that only move work (placement, layout, which core does what), require bit-identical outputs. When
the order of a sum changes (fused reductions) or the math changes (precision), require identical greedy tokens on a
fixed prompt set plus the model's accuracy gate (its own accuracy test against the reference model), and for multi-row
paths a teacher-forced comparison with the previous path.

**Why:** an unchanged output is the cheapest proof that a speedup did not come from skipped work.

## PM-07: Give each op change its own PR and a real-time-profiler perf test

**Do:** keep changes to shared ttnn ops out of model PRs: one PR per op, with the op's existing tests and its nightly
test directory (`tests/ttnn/nightly/unit_tests/operations/<category>/`) passing. Add a perf test that collects device
times with `profile_realtime_program(device, run_fn, collect_all=True)` or `assert_op_duration_merged` from
`tests/ttnn/profiling/realtime_profiler_utils.py`, and asserts either a band around a measured time or the ratio of the
new sequence of programs to the old one (independent of the board's speed). Mark it with the `requires_host_iommu`
pytest marker (the profiler needs a host-IOMMU runner on Blackhole); CI runs marked tests only in its IOMMU job.

**Example:** `ttnn.scatter` (#62558) gates three shapes at ±10% of measured times, which the previous kernel misses by
17–21%. The DRAM-sharded matmul (#62566) requires the slowest chip within 10% of the fastest on a 1×4 mesh, which the
first-chip placement misses by 45%.
