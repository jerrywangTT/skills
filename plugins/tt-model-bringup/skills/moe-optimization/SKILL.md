---
name: moe-optimization
description: "Optimize mixture-of-experts (MoE) layers in TTNN decode and prefill: compute only routed experts, take the union of several rows' experts once, prefetch likely experts, compute only routed (row, expert) pairs for many rows, avoid per-expert token copies and gathers in prefill, keep routing small, and choose the expert kernel by tokens per expert. Use when an MoE layer is a material share of decode or prefill time."
---

# Mixture-of-experts optimization: decode and prefill

Before using this workflow, follow the dependency and environment setup in
[the model-bringup entrypoint](../model-bringup/SKILL.md#startup).
Resolve its package root from this installed skill's location; do not assume
this plugin is copied into the target tt-metal checkout.

A token needs only its k routed experts (out of E). Every byte read or row computed for an expert the token did not
pick, and every copy that only rearranges rows for the experts, is waste. Terms: a *slot* is one of a token's k expert
choices; a *pair* is one (token, expert) choice; the *dispatch* places each pair's token in its expert's rows of an
expert buffer, and the *combine* adds each token's k expert outputs, weighted by the router.

Evidence is from gpt-oss-20b (24 layers, E = 32, k = 4, hidden 2,880) on a QuietBox 2 (4 Blackhole chips):
tenstorrent/tt-metal #60161 (batch-1 decode), #61050 (speculative decoding) and #61548 (prefill, 32–256 users). Prefill
numbers are time to first token, each change measured against the previous one. See also
[Sparse Matmul `nnz`](../optimize/SKILL.md#sparse-matmul-nnz).

## MO-01: Decode: read and compute only the routed experts

**When:** decode runs the expert matmuls over all E experts and zero-weights the unpicked ones.

**Do:** feed the router's expert ids to the expert matmuls so only those experts' weights are read: the stock route is
`ttnn.sparse_matmul` with a sparsity tensor (see the `nnz` rules); a custom streaming kernel can take the ids directly
and also fuse bias, activation and the weighted sum ($decode-weight-streaming WS-01).

**Evidence:** all 32 experts (28 zero-weighted) → only the 4 routed: 275 → 81 µs per layer, 13.77 → 9.34 ms per token.

## MO-02: Several rows: compute the union of their experts once

**When:** a decode step has a few rows (speculative verify rows, a small batch), each picking its own k experts.

**Do:** the router writes each row's top-k, the union of all rows' experts, and a [rows, union] weight table (0 where a
row did not pick an expert). Read each union expert's weights once and multiply by all rows. Send the union to the expert
cores before the weights. The union size, not the row count, sets the bytes read. For R independent rows it averages
E × (1 − (1 − k/E)^R); rows of one sequence (speculative drafts) agree more and give fewer.

**Evidence:** 8 verify rows pick 13.4 of 32 experts per layer on average (21 for independent rows), about 3.4x one
token's expert bytes.

## MO-03: Prefetch the experts the next step will probably pick

**When:** expert weights start streaming only after the router picks, and consecutive steps pick similar experts.

**Do:** while the router runs, read the previous step's most-used experts per layer (as many as fit in the expert cores'
free L1) into L1; use them if picked, and stream the rest from DRAM as before.

**Evidence:** last round's 3 most-used experts per layer were picked again 93.5% of the time: verify 5.78 → 5.49 ms.

## MO-04: Many rows: compute only the routed (row, expert) pairs

**When:** the union (MO-02) is close to all E experts, but each row still needs only k: batched decode with many users,
or prefill.

**Do:** group rows by expert and compute each expert only on its own rows. In prefill, split the experts over the chips
so each chip computes the pairs of its experts, then combine. Make each pass cover enough rows (256–512 here) that its
fixed cost, mainly reading the weights once, is spread thin.

**Evidence:** prefill on routed pairs only: 1K prompt 1,198 → 137 ms. Batched plain decode, 256 users at a 256-token
prompt, one pass with routed-pair MoE: 22.27 → 17.64 ms per step (later work reached 16.1 ms).

## MO-05: Prefill: index instead of copying tokens per expert

**When:** the dispatch copies every token into a [k × tokens] expert buffer before the expert op, and the combine copies
each token's k expert rows back before the weighted sum.

**Do:** the expert op reads token rows by index and writes plain rows; the combine reads each (token, slot) row by index
and takes the router weights as [tokens, k]. In ttnn these are `x_row_index`, `output_row_major` and `zero_output_row`
on `unified_routed_expert_moe`, and `combine_row_index` and `skip_row` on `post_combine_reduce` (proposed in #62564 and
#62565; check the target tt-metal has them). Without them, keep the copies in L1 (MO-06).

**Evidence (8K prompt):** index input 230.7 → 216.5 ms; plain-row output 217.4 → 201.2 ms; indexed combine 191.1 → 180.6
ms.

## MO-06: Prefill: keep routing small and its buffers in L1

**When:** routing (each pair's position in the expert buffer, scatters, gathers) is more than a few percent of an MoE
layer.

**Do:**
- compute pair positions as an exact prefix sum by matmul with a 0/1 triangular matrix built once for all layers; 4-bit
  inputs are exact with fp32 accumulation (`fp32_dest_acc_en`) and an fp32 or integer output; for long chunks C, sum
  within blocks of 512 tokens and then over block totals instead of one [C, C] matrix;
- split a scatter over many cores, each writing its row group into its own zeroed copy of the output, then take the
  max over the copies (each position is written by exactly one core);
- serve the padding row that most gather rows point at from a per-core L1 copy (`padding_idx` of `ttnn.embedding`);
- keep the dispatch input and combine blocks in L1 when they fit;
- assign experts to chips by measured load on text other than the benchmark's.

**Evidence (8K prompt):** running sum on one core → matmul 811 → 586 ms; padding row from L1 477 → 386 ms; experts
assigned by load (busiest chip had up to 1.5x the average tokens) 332 → 314 ms.

## MO-07: Choose the expert kernel by tokens per expert

**When:** the target tt-metal has more than one routed-expert op (`unified_routed_expert_moe`, `moe_fused_swiglu`,
`hybrid_routed_expert_moe`).

**Do:** time each op alone on the model's expert shape, activation and biases over the full range of tokens per expert
(on average prompt length × k ÷ E) before trusting `hybrid_routed_expert_moe`, which picks per expert at run time from a
threshold ($perf-measurement PM-02).

**Evidence:** for the gpt-oss expert shape (2880 × 2880, SwiGLU-OAI with biases),
`models/demos/deepseek_v3_d_p/reference/gpt_oss_20b_config.py` records `moe_fused_swiglu` faster up to 384 tokens per
expert, the unified op at 416–512, `moe_fused_swiglu` again at 576–640 and the unified op from 768. Not yet measured
inside gpt-oss's prefill.
