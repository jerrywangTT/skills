# Fusion card: fewer programs, fewer host waits, fewer DRAM round trips

Read this card when a traced decode step (or a speculative-decoding round) is a long chain of small programs, when
device time per step is well above the sum of the material op rows, or when the host appears between device passes.
The evidence is from gpt-oss-20b on a QuietBox 2 (4 Blackhole chips, TP=4): batch-1 decode in tenstorrent/tt-metal
#60161 and speculative decoding (DFlash) in #61050.

Merging work into one program saves time in three different ways. Name which one a candidate targets, because they
are measured differently:

1. **Overlap.** Inside one program a later stage can start before an earlier one finishes (stream weights while the
   previous stage computes, start attention as soon as Q, K and V are written). Between programs it cannot: the next
   program starts after the previous one ends.
2. **Fixed cost per program.** Every program pays a start and finish cost (kernel binaries and runtime arguments sent
   to every core, cores starting, the end signal) whatever its work. Fewer programs pay it fewer times.
3. **DRAM round trips.** Separate programs usually hand data over through DRAM. One program, or outputs kept in L1,
   does not.

## FU-01: Make each layer boundary one program, and run it inside the next matmul's program

**Applies when** the residual add, the all-reduce of a tensor-parallel matmul's partial sums and the next RMSNorm run
as separate ops between layers.

**Do:** one boundary program per chip that receives the other chips' partial sums over the fabric, adds them, adds
the residual and computes the next norm. Keep the hidden vector as one flat row read in one request, not as many
small tiles. Then run the boundary on spare cores inside the next matmul's program so the matmul's weights stream
while the boundary computes. Have the producing matmul send its partial sum as soon as each piece is written.

**Evidence (#60161, ITL = ms per token at ISL 128):** all-reduce, residual add and RMSNorm as one op on one core per
chip, the vector read in one request: 4.67 → 4.18 ms (96 fewer launches per token); partial sums sent while being
written: 4.18 → 4.12 ms; boundary inside the next matmul's program (10 → 8 ops per layer): 4.11 → 3.98 ms.

## FU-02: Pack the all-reduce payload and use the full fabric

**Applies when** a decode all-reduce sends a 1-row vector padded to whole tiles, or partial sums go between chips in
many small packets, or only one of the links between two chips carries traffic.

**Do:** send the hidden vector as dense tiles (2,880 values: 3 tiles, not 90 tiles with only row 0 used), send
partial sums in large fabric packets, and spread sends over both links.

**Evidence:** 90 tiles with one used row → 3 dense tiles: 8.4 → 3.9 µs per all-reduce (#60161); 46 KB partial sums in
15,232-byte packets: round 7.95 → 7.77 ms; both links for boundary sends: verify 4.29 → 4.07 ms (#61050).

## FU-03: Fold small selection steps into the kernel that produces their input

**Applies when** a router's top-k and softmax, a top-k over logits, or an argmax follows a matmul as separate ops.

**Do:** run them in the producing kernel's writer on the data-movement core, on the output tile while it is in L1.
Use order-preserving integer keys for BF16 values, a running sorted top-k in registers (one comparison against the
k-th best for most values), and 32-bit loads of two values at a time; keep `volatile` off the read pointer after an
explicit cache invalidate so the compiler can batch loads.

**Evidence:** router top-4 + softmax as one op instead of 7: 19.9 → 7.2 µs per layer; then inside the router kernel's
writer: 4.84 → 4.78 ms per token (#60161). Multi-row router (8 rows, top-4 each, their union and the weight table)
rewritten as one pass on word loads: about 14 → 2 µs per layer (#61050).

## FU-04: Run a block's stages as one program with semaphore hand-off

**Applies when** stages of one block (router → expert gate/up → expert down; boundary → MLP; q/k preparation → K/V
cache write) run as consecutive programs on disjoint cores and each waits for the previous program to end.

**Do:** one program with each stage on its own cores. Hand data over through L1 and program semaphores, and let the
later stage's weight readers start as soon as they know which weights to read (for MoE: send the expert list before
the routing weights). Raise the kernel-config buffer if the merged program's binaries and arguments do not fit, or
every launch stalls.

**Evidence (#61050):** verify MoE (router, gate/up, down) as one program, down reading weights while gate/up computes:
verify 6.59 → 6.31 ms; drafter MLP as one program: draft 1.87 → 1.71 ms; drafter q/k preparation and K/V write as one
program instead of 7 ops per layer: draft 1.48 → 1.30 ms; context pass one program for 8 layers instead of 8: round
6.07 → 6.03 ms; attention inside the QKV program: round 6.03 → 5.95 ms; expert list before the routing weights: round
5.087 → 5.075 ms. The merged MoE program needed about 48 KB more kernel-config space per core; without it each launch
stalled about 19 µs.

## FU-05: Keep intermediates and resident data in L1

**Applies when** an op writes an output to DRAM that the next op reads straight back, or a small tensor is re-read
from DRAM every step.

**Do:** keep attention outputs, activations and other single-consumer intermediates in L1. When something else keeps
buffers resident in L1 (a traced runner), size every op's circular buffers below the lowest resident address rather
than assuming all of L1 is free.

**Evidence:** attention outputs and drafter activations kept in L1: draft 1.26 → 1.22 ms, verify 4.66 → 4.61 ms
(#61050). Prefill ops sized for all of L1 crashed with "static circular buffers clash with L1 buffers" once DFlash's
~176 KB per core stayed resident; fitted below it they ran (1K prefill with DFlash loaded: 63 ms, plain 62 ms; #61548).

## FU-06: Launch each kernel's cores as rectangles

**Applies when** a kernel's cores are listed one by one (one `CoreRange` per core), which happens with irregular
reader placements.

**Do:** coalesce the cores into the fewest rectangles (`CoreRangeSet` of maximal ranges). Dispatch sends binaries and
runtime arguments per range.

**Evidence:** cores grouped into rectangles (launch time only): 4.97 → 4.84 ms per token (#60161).

## FU-07: Keep the host off the device's critical path

**Applies when** the host reads results between device passes, computes something the next pass needs, and writes it
back, or a step is several traces.

**Do:** let each pass write the next pass's inputs on the device (positions, last token, accepted count), read
results with non-blocking reads, queue the next step's trace before reading this step's result, and record one trace
per step instead of several.

**Evidence (#61050):** passes hand drafts and inputs over on the device, host reads one 16-integer result: round 16.2 →
10.2 ms; acceptance computed on the device (host 0.3 → 0.07 ms per round): 8.18 → 7.95 ms; next round queued before
reading this one: 4.957 → 4.884 ms; one trace per round instead of three (each trace's first program started
11–16 µs late): 4.873 → 4.842 ms.

## FU-08: Write the decode step's inputs with one kernel

**Applies when** the embedding row, rotary rows and positions for a decode step are produced by several small ops.

**Do:** one kernel writes the embedding row and the Q/K rotary rows for the step's position.

**Evidence:** 10 ops → 1: 16.1 → 1.5 µs, 2.47 → 2.42 ms per token (#60161).
