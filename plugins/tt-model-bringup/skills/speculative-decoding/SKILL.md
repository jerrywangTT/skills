---
name: speculative-decoding
description: "Add or optimize greedy draft-and-verify speculative decoding (a small drafter proposes several tokens, the full model checks them in one pass) for a TTNN model whose plain decode is already optimized. Use when greedy decode is bound by reading weights from DRAM and a drafter trained for the target model exists, when a speculative round is slower than expected, or when serving many users with speculative decoding."
---

# Speculative decoding

Before using this workflow, follow the dependency and environment setup in
[the model-bringup entrypoint](../model-bringup/SKILL.md#startup).
Resolve its package root from this installed skill's location; do not assume
this plugin is copied into the target tt-metal checkout.

**Scope:** greedy decoding. Sampled decoding needs rejection sampling, which this skill does not cover. The drafter
must share the target's tokenizer and be trained for that target. Evidence here uses DFlash (the z-lab gpt-oss-20b
drafter, 8 layers), which proposes all n drafted tokens in one forward pass; an autoregressive drafter costs n passes.

**A round** has three device passes:
1. **Draft:** the drafter proposes n tokens (n = 7 here).
2. **Verify:** the full model processes n + 1 rows in one pass (the last accepted token and the n guesses). Keep the
   guesses up to the first one that differs from the full model's choice, plus the full model's token there.
3. **Context:** write the accepted tokens' drafter inputs into the drafter's K/V cache for the next round.

**Speed = tokens accepted per round ÷ round time.** A verify pass reads every weight once, like one plain decode step,
so a round costs about one plain step plus the drafter and returns several tokens. Kernel changes move the round time;
the drafter and the prompt move the tokens per round. Report both.

Evidence: gpt-oss-20b on a QuietBox 2 (4 Blackhole chips), tenstorrent/tt-metal #61050 and #61548. Plain batch-1 decode
2.38 ms per token (420 tok/s/user); speculative decoding 772 tok/s/user at a 256-token prompt (1.86x), ranging from
1.69x (512 tokens) to 2.10x (64K tokens).

## Before optimizing

- Optimize plain decode first ($decode-weight-streaming, $kernel-fusion, $moe-optimization). The verify pass uses the
  same weights and kernels, so every plain-decode win carries over.
- Check correctness by teacher forcing: feed the speculative output to plain greedy decode and compare its argmax with
  the next speculative token at each position. Computing several rows at once flips near-ties, so require agreement
  within about 0.01 of plain decode checked the same way (gpt-oss: 0.988 at batch 1; 0.968–0.978 at 32 and 128 users,
  the same range as batch-32 plain decode).
- Measure tokens per round over several prompts and prompt lengths. If the chat template stamps the date (gpt-oss
  does), pin it or average over several dates; the date alone changed acceptance between days.

## SD-01: Verify all draft rows in one traced pass

**Do:** one verify pass with n + 1 rows instead of n + 1 decode steps; each weight is read once and applied to all rows,
and the MoE reads the union of the rows' experts ($moe-optimization MO-02). Trace the round.

**Evidence:** first version (eager, one decode step per guess): 180.7 ms per round; one traced 8-row pass: 16.2 ms.
This step changed both batching and tracing.

## SD-02: Keep the round on the device

**Do:** each pass writes the next pass's inputs on the device (drafts into the verify input; accepted count, next
position and last token into the next round); compute the acceptance in the verify pass; queue the next round before
reading this one; one trace per round ($kernel-fusion FU-08).

**Evidence:** hand-over on the device: 16.2 → 10.2 ms per round; acceptance on the device and queuing ahead saved a
further 0.23 and 0.07 ms.

## SD-03: Run the drafter on the plain-decode kernels

**Why:** the drafter is small, so its time is mostly fixed costs: launches, layer boundaries, small attention programs.

**Do:** use the plain decode's streaming matmuls ($decode-weight-streaming WS-01), one-program MLP and boundaries
($kernel-fusion FU-04, FU-01), and a 4-bit drafter LM head.

**Evidence:** streaming kernels and a 4-bit drafter LM head: draft 2.7 → 1.87 ms; MLP as one program: 1.87 → 1.71 ms.

## SD-04: Read the K/V cache once for all rows

**Do:** the n + 1 rows share one history. Stock `ttnn.transformer.paged_scaled_dot_product_attention_decode` reads it
once per row; instead read each K/V chunk once and apply it to all rows (gpt-oss: `tt/dflash/block_attention.py`). For
short sliding windows, a small dedicated kernel (`tt/dflash/row_attention.py`) avoids the stock op's fixed cost.

**Evidence:** 131K-token prompt: 18.01 → 8.20 ms per round; 128-token window layers on the small kernel: 6.97 → 6.66 ms.

## SD-05: Give the drafter enough context

**Do:** the drafter's window is how many recent positions of the target's context it attends to. Sweep it at the prompt
lengths served, using one window size for all lengths within a sweep ($perf-measurement PM-02), over several dates.

**Evidence:** window 512 → 1,024 tokens: same round time; mean 658 → 668 tok/s/user over 5 dates × 10 prompt lengths.

## SD-06: Many users: one pass over every user's block of rows

**Do:** with U users and a block of b rows per user (b ≤ n + 1, including the last accepted token), run draft, verify
and context as passes over U × b rows, accepting on the device. A pass has a fixed cost (reading the weights once) and a
cost per row (all-reduce, LM head, attention, norms, MoE work), so: keep passes at 256 rows; raise to 512 when U × 4 >
256 so that b stays at least 4 where possible; compute only routed (row, expert) pairs ($moe-optimization MO-04).
Compare with plain decode at the same user count and input sequence length (ISL, prompt tokens): at enough users,
speculative decoding loses.

**Evidence (ISL 256, tok/s/user, plain → speculative):** 32 users (b 8, 256-row passes) 105 → 234; 64 users (b 4)
89 → 168; 128 users (b 4, 512-row passes) 79 → 106; 256 users (b 2, 512-row passes) 62 → 58 (it wins only from 4K
prompt tokens). Shorter blocks keep most of the acceptance: 64 users, b = 8 / 4 / 2 accept 4.81 / 3.18 / 1.84 tokens
per round.

## SD-07: Prefill next to a resident drafter

**Do:** see $kernel-fusion FU-05: prefill ops must fit their circular buffers below the drafter's resident L1 buffers.

## Evidence to leave

- Round time split into draft, verify, context and host for each change, with plain decode from the same traced
  harness and build.
- Tokens per round and tok/s/user over several prompts, lengths and (with a dated template) dates.
- Teacher-forced agreement with plain greedy decode.
- For many users: plain vs speculative at each user count and length, with b and rows per pass.
