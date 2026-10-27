---
name: prefill-optimization
description: "Cut time to first token in TTNN prefill beyond matmul configs and chunk sizes: power-of-two padding buckets, attention on every core with per-length chunk sizes, attention math matched to the K/V dtype, biases inside matmuls, rows split over the chips between layers, cheaper collectives, many users in one prefill pass, and fewer small ops. Use when prefill or batched prefill time to first token is a target and its profile shows attention, collectives, per-user passes or small ops as material."
---

# Prefill optimization

Before using this workflow, follow the dependency and environment setup in
[the model-bringup entrypoint](../model-bringup/SKILL.md#startup).
Resolve its package root from this installed skill's location; do not assume
this plugin is copied into the target tt-metal checkout.

Start from [Prefill Latency And TTFT](../optimize/SKILL.md#prefill-latency-and-ttft), which covers 2D matmul configs
and chunk sizes; for mixture-of-experts layers use $moe-optimization. Evidence is time to first token (TTFT) of
gpt-oss-20b on a QuietBox 2 (4 Blackhole chips, TP=4), tenstorrent/tt-metal #61548, at batch 1 unless it says batch 32
(32 users' prompts in one call). Each number pair is one change against its own baseline measured back to back.

## PF-01: Pad prompts to the next power of two

**When:** prompts are padded to a large fixed bucket (gpt-oss padded 129–512 tokens to 1,024).

**Do:** add a bucket at every power of two from the smallest bucket up to the chunk size, and capture each in warmup.

**Evidence:** 256-token prompt 92 → 54 ms; 512 tokens 92 → 67 ms.

## PF-02: Run attention on every core, with chunk sizes chosen per length

**When:** prefill attention uses part of the compute grid, or one chunk size for every prompt length.

**Do:** pass `ttnn.SDPAProgramConfig(compute_with_storage_grid_size=<full grid>, q_chunk_size=..., k_chunk_size=...)`;
sweep the chunk sizes (64–512) at each prompt length and keep the best in a table by length.

**Evidence:** 64 of 110 cores → all: 32K 2,376 → 2,220 ms, 64K 5,738 → 5,122 ms; with per-length chunks from 4K, 128K
15,678 → 12,497 ms.

## PF-03: Match attention math fidelity to the K/V cache dtype

**When:** attention runs HiFi4 math on an 8-bit (BFP8) K/V cache.

**Do:** set `math_fidelity=ttnn.MathFidelity.HiFi2` in the attention's compute kernel config and run the model's accuracy
gate (the model's own accuracy test against its reference). Measure again for other cache dtypes.

**Evidence:** 32K 2,158 → 2,084 ms, 64K 4,733 → 4,437 ms; accuracy gate passes.

## PF-04: Add biases inside the matmul

**When:** a separate op adds a projection's bias after the matmul.

**Do:** use `ttnn.linear(..., bias=...)` instead of a matmul followed by an add.

**Evidence:** the separate o_proj bias pass cost about 20 ms per call at 128K; 8K 488 → 478 ms.

## PF-05: Split rows over the chips between layers

**When:** in tensor-parallel prefill on N chips, every chip runs the residual adds and norms on all rows, and the MoE
router input is gathered from all chips before routing.

**Do:** have each chip keep and process 1/N of the rows between layers (reduce-scatter instead of all-reduce), route
each chip's own rows, and gather only the expert ids and an 8-bit expert input. Measure both layouts over all prompt
lengths; gpt-oss uses the split layout from 1K tokens, where it measured faster.

**Evidence:** rows split: 8K −4.0%, 32K −3.6%; routing on own rows: 8K −3.3%.

## PF-06: Make prefill collectives cheaper

**When:** collectives are material in a prefill profile.

**Do:**
- run a projection on unpadded weights followed by one all-reduce, rather than padded weights followed by
  reduce-scatter + all-gather + slice;
- typecast the attention output to `bfloat8_b` before its all-reduce, if the accuracy gate passes;
- set `num_links` to all links between each chip pair, and tune long reduce-scatters' `chunks_per_sync` (128 here).

**Evidence:** o_proj on unpadded weights with one all-reduce: 1.99 → 1.36 ms per layer at 8K; 8-bit attention
all-reduce: 8K 298 → 284 ms; both links: batch 32 at 4K 4,700 → 4,386 ms.

## PF-07: Prefill many users in one pass

**When:** batched prefill runs one prefill per user, re-reading every weight each time.

**Do:** send all users' tokens through the projections and MoE together and run attention per user, keeping each
user's output bit-identical to its batch-1 prefill. Cap the tokens per pass at what fits in memory and prefill larger
batches in groups (gpt-oss: 64K tokens per pass, so 16 users at 4K and 8 at 8K). Sample the first tokens on the device
in the same pass.

**Evidence:** batch 32 × 128 tokens 1,441 → 165 ms; × 1K 1,993 → 954 ms; groups at 4K 4,421 → 3,771 ms. With on-device
sampling requested, which had forced one prefill per user: 1,334 → 151 ms at 128 tokens.

## PF-08: Remove small copy and reshape ops

**When:** the prefill profile has many small ops that only copy, slice, concatenate or reshape.

**Do:** use zero-copy views and write each output block straight into its rows of one preallocated output; have
attention write its heads already concatenated; give matmuls on few rows (the router at ≤128 tokens) a config whose
`in0_block_w` is the full K in tiles.

**Evidence:** zero-copy views and in-place blocks: batch 32 at 1K 917 → 868 ms; heads concatenated by attention:
8K 182.6 → 179.7 ms; router with one K block: 77.5 → 12.3 µs per layer at 128 tokens.
