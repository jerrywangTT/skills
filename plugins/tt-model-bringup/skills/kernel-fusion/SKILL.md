---
name: kernel-fusion
description: "Cut fixed per-program costs, host waits and DRAM round trips in traced TTNN decode or speculative-decoding steps: one-program layer boundaries inside the next matmul, packed all-reduce payloads, selection inside the producing kernel, one-program blocks with semaphore hand-off, L1 residency, launch rectangles and host-free steps. Use when a step is a long chain of small programs or device time well exceeds the sum of its material op rows."
---

# Kernel fusion: fewer programs, fewer host waits, fewer DRAM round trips

Before using this workflow, follow the dependency and environment setup in
[the model-bringup entrypoint](../model-bringup/SKILL.md#startup).
Resolve its package root from this installed skill's location; do not assume
this plugin is copied into the target tt-metal checkout.

Merging work into one program saves time in three ways. Name which one a change targets and measure it that way:

| Saving | What it is | How to see it |
|---|---|---|
| Overlap | inside one program a later stage starts before an earlier one ends (weights stream while the previous stage computes); between programs it cannot | stage timestamps inside the program (`DeviceZoneScopedN`) |
| Fixed cost per program | binaries and runtime arguments sent to every core, cores starting, the end signal | number of programs per step and op-to-op gaps in `tt-perf-report` |
| DRAM round trip | separate programs hand data over through DRAM | bytes written and read straight back |

Evidence is from gpt-oss-20b on a QuietBox 2 (4 Blackhole chips, TP=4): batch-1 decode (tenstorrent/tt-metal #60161) and
speculative decoding (#61050), where a *round* is one draft pass, one *verify* pass that checks 8 rows, and a context
pass ($speculative-decoding). Most entries need custom kernels launched with `ttnn.generic_op`; reference code is in
`models/demos/gpt_oss/tt/fused_decode/` (`boundary.py`, `stream.py`, `terminal.py`) and `tt/dflash/`.

## FU-01: One program per layer boundary, running inside the next matmul

**When:** the all-reduce of a tensor-parallel matmul's partial sums, the residual add and the next norm run as separate
ops between layers.

**Do:** a custom boundary program per chip receives the other chips' partial sums over the fabric, adds them and the
residual, and computes the norm, with the hidden vector as one flat row read in one request. Run it on spare cores of
the next matmul's program, so that matmul's weights stream meanwhile. Have the producing matmul send each partial sum
as it is written. With several rows (a verify pass), split the vector over a few boundary cores.

**Evidence:** one boundary op (96 fewer launches per token): 4.67 → 4.18 ms per token; inside the next matmul: 4.11 → 3.98
ms; 8-row boundary on 4 cores instead of 1: round 7.48 → 6.97 ms.

## FU-02: Pack the all-reduce payload and use every link

**When:** a decode all-reduce sends a one-row vector as one-row tiles, sends in small packets, or uses one of the links
between two chips.

**Do:** pack the H values into ceil(H / 1024) dense tiles instead of H / 32 one-row tiles (gpt-oss H = 2,880: 3 tiles,
not 90). Raise the fabric packet size before opening the mesh: `cfg = ttnn.FabricRouterConfig();
cfg.max_packet_payload_size_bytes = 15232`. Set the collective's `num_links` to all links between each chip pair (2 on a
QuietBox 2).

**Evidence:** packed payload 8.4 → 3.9 µs per all-reduce; 15,232-byte packets: round 7.95 → 7.77 ms; both links: verify
pass 4.29 → 4.07 ms.

## FU-03: Do small selections inside the kernel that produces their input

**When:** a router's top-k and softmax, a top-k or an argmax follows a matmul as separate ops.

**Do:** run them in the producing kernel's writer while the output tile is in L1. Compare BF16 values as integer keys
(flip the sign bit of positives, invert all bits of negatives, so integer order is float order), keep a running top-k in
registers (one comparison against the k-th best rejects most values), and load two values per 32-bit word.

**Evidence:** router top-4 + softmax, 7 ops → 1: 19.9 → 7.2 µs per layer; 8-row router with the union of experts:
about 14 → 2 µs per layer.

## FU-04: Run a block's stages as one program with semaphore hand-off

**When:** stages of one block (router → expert gate/up → down; boundary → MLP; q/k preparation → cache write) run as
consecutive programs on separate cores.

**Do:** one program with each stage on its own cores; hand data over through L1 and semaphores; start the later
stage's weight reads as soon as it knows which weights (send the expert list before the routing weights). If the merged
program's binaries and arguments exceed the kernel-config buffer, every launch stalls: enlarge it by lowering
`worker_l1_size` when opening the device (gpt-oss: Blackhole's 1,461,120 bytes minus 48 KB).

**Evidence:** verify MoE as one program, down reading while gate/up computes: 6.59 → 6.31 ms; drafter q/k preparation
and cache write, 7 ops per layer → 1: 1.48 → 1.30 ms. Without the larger buffer each launch stalled about 19 µs.

## FU-05: Keep intermediates and resident data in L1

**When:** an op writes an output to DRAM that the next op reads straight back.

**Do:** keep single-consumer intermediates in L1. When a traced runner keeps buffers resident in L1, read the lowest
resident address and pass it to the other ops so their circular buffers end below it (block sizes chosen to fit), as
the `l1_end` option of `unified_routed_expert_moe` does (proposed in #62564).

**Evidence:** attention outputs and drafter activations in L1: verify 4.66 → 4.61 ms. Prefill ops sized for all of L1
failed with "static circular buffers clash with L1 buffers" next to ~176 KB per core of resident buffers; fitted below
them, a 1K prefill ran in 63 ms (62 ms with no resident buffers).

## FU-06: Spread a shared input instead of having every core fetch it

**When:** many cores of one program each read the same input from one core (every o_proj core gathering all attention
rows; 24 down-projection cores reading their input from one core).

**Do:** each core reads 1/n of the input and forwards it to the others (or multicasts it), so the source core serves
each byte once.

**Evidence:** shared o_proj input gather: round 8.32 → 8.18 ms; MoE input spread over cores: 6.48 → 6.40 ms.

## FU-07: Launch each kernel's cores as rectangles

**When:** a kernel's cores are listed one `CoreRange` per core, as irregular reader placements produce.

**Do:** coalesce the cores into the fewest rectangles before building the program (by hand, as `_BankStreamOp` in
`stream.py` does); dispatch sends binaries and arguments per range.

**Evidence:** 4.97 → 4.84 ms per token.

## FU-08: Keep the host off the device's critical path

**When:** the host reads results between device passes, computes the next pass's inputs and writes them back, or one
step is several traces.

**Do:** let each pass write the next pass's inputs on the device. Replay with `ttnn.execute_trace(..., blocking=False)`,
start the result read with `tensor.cpu(blocking=False)` and `ttnn.record_event`, queue the next step's trace, then wait
with `ttnn.event_synchronize` and read. Record one trace per step: each trace's first program started 11–16 µs late.

**Evidence (rounds):** inputs handed over on the device: 16.2 → 10.2 ms; acceptance on the device (host 0.3 → 0.07 ms):
8.18 → 7.95 ms; next round queued first: 4.957 → 4.884 ms.

## FU-09: Write a decode step's inputs with one kernel

**When:** the embedding row, rotary rows and positions are produced by several small ops each step.

**Do:** one kernel writes them for the step's position.

**Evidence:** 10 ops → 1: 16.1 → 1.5 µs.
