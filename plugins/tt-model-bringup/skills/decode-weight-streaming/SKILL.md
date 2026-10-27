---
name: decode-weight-streaming
description: "Make decode matmuls that are bound by reading weights from DRAM run near DRAM bandwidth on Tenstorrent hardware: stream weights in large reads from contiguous per-bank columns, place readers on their bank's row, balance banks, place readers per chip on harvested multi-chip meshes, and pick LM-head top tokens inside the LM-head program. Use when a decode QKV, attention-output, router, expert or LM-head matmul takes more than about 1.3x its weight-bytes-over-bandwidth floor."
---

# Decode weight streaming

Before using this workflow, follow the dependency and environment setup in
[the model-bringup entrypoint](../model-bringup/SKILL.md#startup).
Resolve its package root from this installed skill's location; do not assume
this plugin is copied into the target tt-metal checkout.

Batch-1 decode reads every weight it uses once per token and computes little on it, so a matmul's floor is its weight
bytes divided by DRAM bandwidth (512 GB/s per Blackhole P150 chip; 490–508 GB/s measured). Compare each material
matmul row in `tt-perf-report` (one at least about 2% of device time) with that floor. Above about 1.3x the floor, the
time goes into how the weights are read. Sweep the stock DRAM-sharded matmul first
([OPT-004](../optimize/SKILL.md#opt-004-sweep-dram-sharded-decode-matmul-geometry-not-only-dtype),
[OPT-015](../optimize/SKILL.md#opt-015-sweep-blackhole-dram-readers-per-bank)); the entries below took gpt-oss-20b on a
QuietBox 2 (4 Blackhole chips, TP=4) from 18.35 to 2.38 ms per token after that (tenstorrent/tt-metal #60161, #61050).
Reference kernels: `models/demos/gpt_oss/tt/fused_decode/kernels/stream_*_reader.cpp` and
`tt/fused_decode/stream.py` in tt-metal.

## WS-01: Stream weights from contiguous per-bank columns in large reads

**When:** a decode matmul stays above about 1.3x its floor after its config, reader count and dtype are swept. The stock
DRAM-sharded matmul issues one read per weight tile (576 bytes in BFP4, 1,088 in BFP8), which caps the rate.

**Do:** at load time, store each bank's share so that the K tiles of each output-column block are consecutive in DRAM
(a cached weight layout, built once). A custom reader kernel reads each block in one request of several KB (7–14 KB
here: as many whole tiles as fit a NoC packet) and keeps the next block in flight (`noc_async_read` with transaction
IDs, waiting only on the oldest). Put the bias, activation (SwiGLU) and, for the MoE down projection, the weighted sum
over experts in the same kernel so the output is written once.

**Check:** the row's effective bandwidth (weight bytes ÷ device time); tokens identical to the stock path on a fixed
prompt set.

**Evidence:** expert gate/up 54 → 23 µs per layer (one tile per read reached about 220 GB/s); LM head 1,218 → 307 µs per
token (158 → 508 GB/s).

## WS-02: Put each reader on its bank's own row of cores

**When:** a bank's readers sit on more than one row of the worker grid.

**Do:** a bank's NoC-0-optimal worker is the core returned for it by
`get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, coord)`. Place the bank's readers on that core's row,
nearest it first. Two banks may share a row: cores west of the second bank's worker serve the first bank, cores from it
eastward serve the second (WS-03 balances the two). Take extra cores from the rest of the row before another row.

**Why:** readers on a neighbouring row lose NoC arbitration to the row's own readers, finish late, and the bank waits for
its slowest reader.

**Evidence:** 464 GB/s per chip with readers spread over neighbouring rows, 490 GB/s on the bank's row.

## WS-03: Balance bytes per bank and keep work off slow banks and cores

**When:** in three or more profiles, the same bank or the same cores finish more than about 10% after the median.

**Do:** count the weight tiles each bank holds and move columns until the busiest holds as few as possible. If the width
is padded to divide over the banks, put the inert padding columns on the slow bank. Give no work to cores that are
always late. Give fewer columns to a bank that shares its reader row (WS-02).

**Check:** per-core end times inside the program: device-profiler zones (`DeviceZoneScopedN`) in the kernels.

**Evidence:** bank 0 ran 15–20% behind; padding moved onto it and the late cores given no work: MoE 122 → 113 µs per
layer.

## WS-04: Place readers by each chip's own layout on a mesh

**When:** one program runs on several Blackhole chips and `get_optimal_dram_bank_to_logical_worker_assignment(noc, coord)`
returns different cores for different chips (they are harvested differently). Check the target tt-metal has this
per-chip overload; the mesh-level `(noc)` form answers for the first chip only and is deprecated for that reason.

**Why:** a chip's best reader for a bank is the first working column east of the bank's DRAM column. Logical columns skip
harvested ones, so the first chip's column can sit west of the bank on another chip, where NoC 0 carries the data round
the whole row. That chip is slower and every collective after the matmul waits for it.

**Do:** build one program per chip with its own placement. If one program must serve all chips: for each bank take the
largest logical column any chip uses as its worker (east of the bank on every chip), add extra readers only east of it
for east-side banks, and keep west-side banks' readers west of every chip's east-side workers.

**Check:** time the same op on every chip (the `DEVICE ID` column of the device-profiler ops report); the slowest within 10% of the fastest.

**Evidence:** per-chip placement 2.335 ms per token, first-chip placement 2.799 ms (two chips' streaming matmuls
1.13–1.34x slower). DRAM-sharded matmul with two readers per bank on those two chips: 216 µs with first-chip placement,
149 µs with per-chip placement (#62566).

## WS-05: Pick the LM head's top tokens inside the LM-head program

**When:** only the top token or a top-k of the LM-head output is used (see also
[LM Head And Sampling](../optimize/SKILL.md#lm-head-and-sampling) and $kernel-fusion FU-03).

**Do:** each core keeps its best candidates of the logits it computed (top-32 for sampling); the chips exchange them
over the fabric from the writer kernel in the same program; one core picks. When several rows are scored (speculative
verify), first read a 4-bit copy of the head and keep each core's 8 best per row, then rescore only those at 8 bits. The
4-bit copy costs about vocab × hidden × 0.56 bytes of DRAM.

**Check:** chosen tokens identical to the full-precision head on a fixed prompt set.

**Evidence:** top-32 kept and exchanged in the program instead of 6 ops after it: 370 → 321 µs; two-stage head for 8
rows: verify pass about 70 µs faster, same tokens.

## WS-06: Use more readers per bank when one cannot keep it busy

**When:** a streaming kernel reads a bank below its bandwidth and the bank's row has spare cores.

**Do:** in your own streaming kernel, sweep 1–4 readers per bank per projection, each reading a disjoint slice; keep each
read large (WS-01). The stock DRAM-sharded matmul allows 1–3 (OPT-015).

**Evidence:** attention output projection 1 → 2 readers per bank: speculative round 6.40 → 6.32 ms.
