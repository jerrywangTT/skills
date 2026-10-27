---
name: speculative-decoding
description: "Add or optimize draft-and-verify speculative decoding (a small drafter proposes several tokens, the full model checks them in one pass) for a TTNN model whose plain decode is already optimized. Use when greedy decode is bound by reading weights from DRAM and a compatible drafter exists, when a speculative round is slower than expected, or when serving many users with speculative decoding."
---

# Speculative decoding

Before using this workflow, follow the dependency and environment setup in
[the model-bringup entrypoint](../model-bringup/SKILL.md#startup).
Resolve its package root from this installed skill's location; do not assume
this plugin is copied into the target tt-metal checkout.

Evidence below is from DFlash (the z-lab gpt-oss-20b DFlash drafter, 8 layers, 7 drafted tokens) on gpt-oss-20b on a
QuietBox 2 (4 Blackhole chips, TP=4): tenstorrent/tt-metal #61050 (batch 1) and #61548 (32–256 users). Plain batch-1
decode there was already 2.38 ms per token (420 tok/s/user); speculative decoding reached 772 tok/s/user at a 256-token
prompt (mean over 5 prompt dates), 1.69–2.10x plain decode from 256 to 128K tokens of prompt.

## How a round works and what sets its speed

1. **Draft:** the drafter proposes the next n tokens (n = 7 here) from the last accepted token and its own context.
2. **Verify:** the full model processes n + 1 rows in one pass: the last accepted token and the n guesses. For each row
   it computes its own next token.
3. **Accept:** keep the guesses up to the first one that differs from the full model's choice, plus the full model's
   token there. Greedy output equals plain greedy decode apart from near-ties.
4. **Context:** write the accepted tokens' drafter features into the drafter's cache for the next round.

Speed = tokens accepted per round ÷ round time. Plain decode reads every weight once per token; a verify pass also
reads every weight once, so a round costs roughly one plain step plus the drafter, and returns several tokens.
Measure both factors separately and report them: optimizations change the round time, the drafter and prompt change
the tokens per round.

## Gates before optimizing

- Optimize plain decode first ($decode-weight-streaming, $kernel-fusion and $moe-optimization). The verify pass runs
  the same weights through the same
  kernels; every plain-decode win carries over.
- Check correctness by teacher forcing: feed the speculative output into plain greedy decode and compare its argmax
  with the next speculative token at every position. Expect high but not perfect agreement, because verify computes
  several rows at once and near-ties flip: 0.988 at batch 1; 0.968–0.978 at 32 and 128 users, the same range as
  batch-32 plain decode checked the same way.
- Measure tokens per round over several prompts and prompt lengths. If the chat template stamps the date into the
  prompt (gpt-oss does), pin it or average over several dates: the date alone moved acceptance between runs.

## SD-01: Verify all draft rows in one pass

Checking the n guesses with n + 1 ordinary decode steps reads every weight n + 1 times per round. One verify pass with
n + 1 rows reads each weight once and multiplies it by all rows; its MoE reads the union of the rows' experts
([SM-02](../moe-optimization/SKILL.md#sm-02-several-rows-at-once-compute-the-union-of-their-experts-once-each)).

Evidence: first version (eager, one decode step per guess) 180.7 ms per round; one traced 8-row verify pass
16.2 ms per round (draft 97.0 → 5.8 ms, verify 77.2 → 8.5 ms).

## SD-02: Keep the round on the device

Let each pass write the next pass's inputs on the device: drafts into the verify input, the accepted count, next
position and last token into the next draft and verify inputs, and the context rows into the drafter. Compute the
acceptance in the verify pass. Read one small result per round with a non-blocking read, queue the next round
before reading this one, and record the round as one trace
([FU-07](../kernel-fusion/SKILL.md#fu-07-keep-the-host-off-the-devices-critical-path)).

Evidence: hand-over on the device, host reads one 16-integer result: 16.2 → 10.2 ms per round; acceptance on the
device: 8.18 → 7.95 ms; next round queued first: 4.957 → 4.884 ms; one trace instead of three: 4.873 → 4.842 ms.

## SD-03: Run the drafter on the plain-decode kernels

The drafter is a small model, so its time is mostly fixed costs: launches, boundaries between layers, small attention
programs. Give it the streaming matmul kernels, one-program MLP and boundaries of the plain decode, a low-precision
drafter LM head, and its K/V written straight into its cache.

Evidence: drafter on the streaming kernels with a 4-bit LM head: draft 2.7 → 1.87 ms; MLP as one program: 1.87 →
1.71 ms; q/k preparation and cache write as one program: 1.48 → 1.30 ms; attention patching the new K/V rows into
its own copy instead of waiting for the cache write: round 5.098 → 5.087 ms.

## SD-04: Read the K/V cache once for all rows

Stock decode attention reads a user's whole K/V cache once per row. The n + 1 verify rows (and the drafter's rows)
share the same history: read each K/V chunk once and apply it to all rows. For short sliding windows, a small
dedicated kernel avoids the stock op's fixed cost; start attention inside the QKV program as soon as Q, K and V exist.

Evidence: shared K/V reads ("block attention"): 131K-token prompt 18.01 → 8.20 ms per round; own 8-row attention for
the 128-token window layers: 6.97 → 6.66 ms; attention inside the QKV program: 6.03 → 5.95 ms.

## SD-05: Give the drafter enough context

A drafter that sees a window of the prompt drafts worse when the window is shorter than the prompt. Sweep the window
at the prompt lengths you serve; one sweep per window setting, all lengths and several prompt dates.

Evidence: window 512 → 1,024 tokens: round time unchanged (4.842 ms), mean over 5 dates × 10 prompt lengths 658 →
668 tok/s/user.

## SD-06: Many users: one pass over every user's block of rows

With U users and a block of b rows per user (the last token plus b − 1 guesses), run draft, verify and context as
passes over U × b rows, with acceptance on the device. A pass has a fixed cost (mostly reading the weights once) and a
per-row cost (all-reduce, LM head, attention, norms, and MoE gather/combine growing with rows × slots). Choose b and the
rows per pass per user count:

- take the largest b (up to the drafter's n + 1) that keeps every user in one pass;
- allow larger passes (512 rows) once users × b would otherwise force small blocks or several passes;
- compute only the routed (row, expert) pairs in the verify MoE
  ([SM-04](../moe-optimization/SKILL.md#sm-04-many-rows-batch-32256-or-prefill-compute-only-the-routed-row-expert-pairs)).

Measure against plain decode at the same user count and prompt length: at enough users the per-row costs make a
speculative round slower than the plain steps it replaces.

Evidence (tok/s/user, ISL 256, plain → speculative): 32 users (8-row blocks, 256-row passes) 105 → 234; 64 users
(4-row, 256) 89 → 168; 128 users (4-row, 512) 79 → 106; 256 users (2-row, 512) 62 → 58, and speculative wins only from
4K tokens of prompt (35 → 37). At 64 users, 8-row blocks in one 512-row pass (155) lost to 4-row blocks in 256-row
passes (168).

## SD-07: Prefill next to a resident drafter

A traced speculative runner keeps buffers resident in L1 between rounds. Prefill ops that size their circular buffers
for all of L1 then fail with "static circular buffers clash with L1 buffers". Size them below the lowest resident L1
address ([FU-05](../kernel-fusion/SKILL.md#fu-05-keep-intermediates-and-resident-data-in-l1)).

## Evidence to leave

- Round time split into draft, verify, context and host, per change, with the plain decode baseline from the same
  traced harness and build.
- Tokens per round and tok/s/user over several prompts, prompt lengths and (if the template is dated) dates.
- Teacher-forced agreement with plain greedy decode.
- For many users: plain vs speculative at each user count and prompt length served, with b and rows per pass.
