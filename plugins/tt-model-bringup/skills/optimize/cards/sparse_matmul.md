# Sparse matmul card: mixture-of-experts decode and prefill

Read this card when a mixture-of-experts (MoE) layer is a material share of decode or prefill time. It complements
[Sparse Matmul `nnz`](../SKILL.md#sparse-matmul-nnz). Evidence is from gpt-oss-20b (24 layers, 32 experts, top-4,
hidden 2,880) on a QuietBox 2 (4 Blackhole chips, TP=4 for decode, experts split over chips for prefill):
tenstorrent/tt-metal #60161 (batch-1 decode), #61050 (speculative decoding) and #61548 (prefill and batched
speculative decoding).

The rule behind every entry: an MoE token needs only its k routed experts. Every byte read or row computed for
an expert the token did not pick, and every copy that only rearranges rows for the experts, is waste.

## SM-01: Decode: read and compute only the routed experts' weights

**Applies when** decode runs the expert matmuls over all experts and masks or zero-weights the unpicked ones.

**Do:** pass the router's expert ids to the expert kernels and read only those experts' weights from DRAM. With one
token, k experts mean k weight reads per layer, not E. Put the bias, activation and the weighted sum of the k experts
in the same streaming kernels (see [MM-01](matmul.md#mm-01-stream-decode-matmul-weights-from-contiguous-per-bank-columns-in-large-reads)).

**Evidence:** expert matmuls stepped through all 32 experts and wrote 28 rows of zeros; taking the 4 ids and writing
4 rows: experts 275 → 81 µs per layer, 13.77 → 9.34 ms per token, the second-largest decode win (#60161).

## SM-02: Several rows at once: compute the union of their experts, once each

**Applies when** decode processes several rows per step (speculative-decoding verify rows, or a small batch) and
each row picks its own k experts.

**Do:** the router writes each row's top-k, the union of all rows' experts, and a [rows, union] weight table (0 where
a row did not pick the expert). Each union expert's weights are read once and multiplied by all rows; the weighted
sum uses the table. Send the union to the expert cores before the weights so their readers start early. Record how
large the union is: it, not the row count, sets the bytes read.

**Evidence:** 8 verify rows pick on average 13.4 of 32 experts per layer, so a verify pass reads about 3.4x one
token's expert bytes (#61050 speed-of-light accounting). Verify MoE as one program (router, gate/up, down): verify
6.59 → 6.31 ms per round.

## SM-03: Prefetch the experts the next step will probably pick

**Applies when** expert weights start streaming only after the router has picked them, and consecutive steps pick
similar experts.

**Do:** while the router runs, read the experts most used in the previous step (per layer) into the expert cores' L1.
Use them if the router picks them; read the rest after.

**Evidence:** last round's 3 most-used experts per layer, read during the router, were picked again 93.5% of the time:
verify 5.78 → 5.49 ms per round (#61050).

## SM-04: Many rows (batch 32–256, or prefill): compute only the routed (row, expert) pairs

**Applies when** the row count is large enough that most experts are picked by some row, so a union (SM-02) means
nearly dense work, but each row still needs only k experts.

**Do:** group the rows by expert (a slot list per expert), and compute each expert only on its own rows. In prefill,
split the experts over the chips so each chip computes the pairs of its experts, then combine. Size the passes so the
per-pass fixed cost (weight reads, launches) is amortized over many rows: for batched speculative decoding, 256 rows
per pass below 128 users and 512 from 128 users.

**Evidence:** prefill, every token through all 32 experts → each chip computes only the pairs routed to its 8
experts: batch-1 time to first token 1K 1,198 → 137 ms, 8K 9,553 → 809 ms (#61548). Batched speculative decoding at
128 users, ISL 256: 2-row blocks in 256-row passes 93 → 4-row blocks in 512-row passes 106 tok/s/user (plain decode
79); at 256 users 512-row passes lift it 47 → 58, still below plain decode's 62.

## SM-05: Prefill: avoid copying each token once per expert

**Applies when** a dispatch gather copies every token into a [k × tokens] expert buffer before the expert op reads it,
and a combine gather copies each token's k expert rows back before the weighted sum.

**Do:** let the expert op read token rows through an index (`x_row_index` on `unified_routed_expert_moe`) and write
its output as plain rows (`output_row_major`, with the spare row the combine points other chips' slots at zeroed by
`zero_output_row`). Let the combine read each slot's row of the expert output through an index
(`combine_row_index`, `skip_row` on `post_combine_reduce`) and the router weights as [tokens, k]. Check these options
exist in the target tt-metal (they are proposed in #62564 and #62565); otherwise keep the gathers in L1 (SM-06).

**Evidence (batch-1 time to first token, 8K):** read by index 230.7 → 216.5 ms; rows written directly 217.4 → 201.2 ms
(32K 939.0 → 873.7); weights as [tokens, k] 199.1 → 196.5 ms; combine by index 191.1 → 180.6 ms (32K 834.8 → 794.7)
(#61548). As isolated sequences on one chip at 2,048 tokens: expert path 1,221 → 1,122 µs, combine path 224 → 91 µs.

## SM-06: Prefill: keep routing small and its buffers in L1

**Applies when** routing (positions of each pair in the expert buffer, scatters, gathers) costs more than a few
percent of an MoE layer.

**Do:**
- compute each pair's position as an exact prefix sum with a matmul against a 0/1 triangular matrix (4-bit is still
  exact), in blocks rather than one [C, C] matrix, and share the matrix across layers;
- run scatters on many cores (one row group per core, merged with a max);
- serve the padding row that most gather rows point at from a per-core L1 copy (`padding_idx`);
- keep the dispatch input and the combine block in L1 when they fit, capped below any L1 other buffers keep resident;
- place experts on chips by measured load on representative text (not the benchmark text), not consecutively.

**Evidence (#61548, batch-1 time to first token):** running sum on one core → one matmul: 8K 811 → 586 ms; scatter on
32 cores: 8K 377 → 350 ms; padding row from L1: 8K 477 → 386 ms; combine in L1 blocks: 8K 350 → 338 ms; 4-bit routing
matrix: 0.375 → 0.234 ms per layer at 8K; experts placed by calibrated load (busiest chip from up to 1.5x average):
8K 332 → 314 ms; one routing matrix shared by all layers: +1,685 → +121 MB DRAM after an 8K prefill.

## SM-07: Choose the expert kernel by tokens per expert

**Applies when** the target tt-metal offers more than one routed-expert op (`unified_routed_expert_moe`,
`moe_fused_swiglu`, `hybrid_routed_expert_moe`).

**Do:** measure both ops on the model's expert shape, with its activation and biases, across tokens per expert
(prompt length × k / experts on average). Few tokens per expert favours `moe_fused_swiglu`; many favours the unified
op. `hybrid_routed_expert_moe` picks per expert at run time from a threshold.

**Evidence:** the DeepSeek prefill team measured the gpt-oss expert shape (2880 × 2880, SwiGLU-OAI with biases):
`moe_fused_swiglu` faster up to 384 tokens per expert, the unified op from 768
(`ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD_MEASURED` in tt-metal's `gpt_oss_20b_config.py`). Not yet measured inside
gpt-oss's own prefill.
