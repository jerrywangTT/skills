# Matmul card: decode weight streaming and DRAM reader placement

Read this card when a decode matmul (QKV, attention output, router, expert gate/up or down, LM head) is a
material row in `tt-perf-report` and decode is bound by reading its weights from DRAM. It extends
[OPT-004](../SKILL.md#opt-004-sweep-dram-sharded-decode-matmul-geometry-not-only-dtype) and
[OPT-015](../SKILL.md#opt-015-sweep-blackhole-dram-readers-per-bank) with what moved gpt-oss-20b batch-1 decode on a
QuietBox 2 (4 Blackhole chips, TP=4) from 18.35 to 2.38 ms per token (tenstorrent/tt-metal #60161) and batch-32
decode from 15.44 to 12.72 ms per step (#61548, #62566). The numbers are evidence for when a rule pays, not targets
for another model.

Start with the arithmetic. Decode at batch 1 reads every weight it uses once per token and computes very little on
it, so a projection's floor is its weight bytes divided by the chip's DRAM bandwidth (512 GB/s per Blackhole P150
chip; about 490 GB/s is what all cores reading together reached). Compare each matmul row's device time with that
floor before changing anything: a row far above it is losing time to how the weights are read, not to math.

## MM-01: Stream decode matmul weights from contiguous per-bank columns in large reads

**Applies when** a DRAM-sharded decode matmul stays well below DRAM bandwidth after its program config, reader count
and dtype are swept. The stock DRAM-sharded matmul requests one weight tile per DRAM read; a 576-byte BFP4 tile or a
1,088-byte BFP8 tile per request caps the read rate far below the bank's bandwidth.

**Do:** store each DRAM bank's share of the weight as contiguous column blocks at load time (a weight-cache layout,
built once), and read it with your own reader kernel in large pieces (several KB per request, as many tiles as fit a
NoC packet) with several reads in flight per reader (transaction-id ring). Fuse what follows the matmul into the same
kernel: bias add, activation (SwiGLU), and for MoE down the weighted sum of the experts, so the output is written once.

**Verify:** the profiler row's effective bandwidth (weight bytes / device time) and the generated text or PCC
against the previous path. Keep the stock op as the control in the same traced harness.

**Evidence (gpt-oss-20b, QuietBox 2, #60161):**
- expert gate/up: one 576-byte tile per request, about 220 GB/s; streamed in 7,488-byte pieces with bias and SwiGLU
  inside: 54 → 23 µs per layer;
- expert down: a zero-fill, a matmul and a sum op became one pass with one request per 14,400-byte column:
  27.6 → 13.1 µs per layer;
- LM head (200 MB per chip): one 1,088-byte tile per request at 158 GB/s; 14 KB pieces at about 508 GB/s:
  1,218 → 307 µs per token.

## MM-02: Put each DRAM reader on its bank's own row of cores

**Applies when** a matmul's readers for one bank sit on more than one row of the worker grid, or on a row that other
banks' readers also use.

**Do:** place every reader of a bank on the row of that bank's NoC-0-optimal worker
(`get_optimal_dram_bank_to_logical_worker_assignment`), nearest that worker first. When two banks share a row,
split the row at the second bank's worker. If a bank needs more cores than its row segment has, take them from the
rest of its row before any other row.

**Why:** readers on a neighbouring row lose NoC arbitration to the row's own readers, finish their share late, and
the bank finishes with its slowest reader.

**Evidence:** all readers streaming one bank's share per reader: 464 GB/s per chip with readers spread over
neighbouring rows, 490 GB/s (95.7% of 512) with every reader on its bank's row (#61050).

## MM-03: Balance the bytes per bank, and keep work off slow banks and cores

**Applies when** one bank or a few cores always finish last in a profile of a streaming matmul, or when a weight's
width does not divide evenly over the banks.

**Do:** count the tiles each bank holds per expert or projection and move columns so the busiest bank holds as
little as possible. Put padding columns on a bank that measures slow. Give no work to cores that measure late every
time. When two banks share one row of readers, give the bank whose readers are short of cores fewer columns.

**Verify:** per-core end times inside the program (device timestamps), not only the program's total.

**Evidence (#61050):** bank 0's readers ran 15–20% behind; padding moved onto bank 0 and the slow cores given no work:
MoE 122 → 113 µs per layer. Expert-down busiest bank 846 → 810 tiles per expert: round 4.997 → 4.954 ms. Banks 0 and
4 share a row of reader cores and bank 4 was the slowest on every chip; fewer columns on bank 4: 4.954 → 4.874 ms.

## MM-04: Place DRAM readers by each chip's own layout on a mesh

**Applies when** one program runs on several Blackhole chips that are harvested differently (most multi-chip
Blackhole boxes; check `harvest_mask` per chip), and a placement is computed once for the mesh.

**Why:** a chip's best reader for a bank is the first unharvested worker column physically east of the bank's DRAM
column (`tt_metal/common/core_assignment.cpp`). Logical columns skip harvested columns, so the same logical column is
a different physical column on each chip. The mesh-level
`MeshDevice::get_optimal_dram_bank_to_logical_worker_assignment(noc)` answers for the first chip only and is
deprecated for that reason. On another chip that column can sit west of the bank's DRAM column, where NoC 0 (east and
south only, with wraparound) carries the bank's data round the whole row, through the other DRAM column, sharing its
links. That chip runs slower, and every collective after the matmul waits for it.

**Do:** query the per-chip assignment (`get_optimal_dram_bank_to_logical_worker_assignment(noc, coord)`). Either build
one program per chip with its own placement, or use per bank the largest logical column over the chips (it is at or
east of the bank on every chip) and grow extra readers only away from the DRAM column (east for east-side banks, and
west-side banks' readers stay west of every chip's east-bank worker).

**Verify:** profile each chip separately: the same op on chips with the same work must take the same time. With the
real-time profiler, record each chip's duration for the op and require the slowest within a few percent of the
fastest. A whole-model average hides this.

**Evidence:**
- QuietBox 2: east banks' best reader is logical column 6 / 7 / 5 / 7 on chips 0 / 1 / 2 / 3. With chip 0's list on
  all chips, batch-1 decode 2.335 → 2.799 ms per token, and on chips 1 and 3 every streaming matmul ran 1.13–1.34x
  slower; chip 2's list gave 2.997 ms.
- DRAM-sharded matmul `[32, 4096] x [4096, 16384]` bf8, two readers per bank: chips 1 and 3 216 µs vs 149 µs with the
  first chip's placement, all four 149 µs with per-chip placement (#62566); batch-32 decode 15.44 → 12.72 ms per step.

## MM-05: Find the LM head's top tokens in two precisions, inside the LM head program

**Applies when** the LM head reads a large vocabulary (hundreds of MB per chip) for every row and only the best token
or a top-k list is needed.

**Do:** keep each core's best candidates inside the LM head program (top-32 for sampling), exchange the lists between
chips in the same program, and pick the token there. For verifying several rows (speculative decoding), read a 4-bit
copy of the head first, keep each core's 8 best candidates per row, and rescore only those at 8 bits. Scan candidates
two 16-bit values per 32-bit word.

**Verify:** the chosen tokens are identical to the full-precision head on a fixed prompt set.

**Evidence:** top-32 kept in the LM head program and exchanged in it: 6 separate ops after the LM head removed,
370 → 321 µs; sampling on 32 candidates: 530 → 65 µs (#60161). Two-stage head for 8 verify rows: round 6.32 → 6.25 ms,
same tokens; two values per word in the scan: 217 → 178 µs per head (#61050).

## MM-06: Read more of a bank with more readers when one cannot keep it busy

**Applies when** a streaming kernel's reader rate per bank is below the bank's bandwidth and spare cores exist on the
bank's row (see [OPT-015](../SKILL.md#opt-015-sweep-blackhole-dram-readers-per-bank) for the stock matmul's
`num_workers_per_dram_bank`).

**Do:** sweep 1–4 readers per bank per projection, each reading a disjoint slice of the bank's columns, under the same
traced harness. Keep the per-reader request size large (MM-01); more readers with tiny requests can lose.

**Evidence (#61050):** attention output projection 1 → 2 readers per bank: round 6.40 → 6.32 ms; 2 → 3: 5.007 →
4.997 ms. Drafter context pass 4 readers per bank: 5.026 → 5.007 ms. Dense-MoE batch-32 decode uses 3 readers per
bank in the stock DRAM-sharded matmul.
