# Submission Compliance Checklist

This document is the working compliance standard for this project.

Primary references:
- Issue `#1017`: [A Field Guide to Valid Submissions](https://github.com/openai/parameter-golf/issues/1017)
- Example writeup to emulate: [April 9 SP8192 leaderboard README](/Users/ifengwu/Projects/parameter-golf/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md)

Use this checklist before trusting any result, before promoting a candidate to cloud, and before writing a submission README.

## Core stance

- Treat `#1017` as the unofficial but practical interpretation of what makes `val_bpb` meaningful.
- Every experiment must declare whether it is targeting `Track A` or `Track B`.
- Any suspiciously large gain is presumed to be a bug or rules violation until checked against this list.

## Track declaration

Choose one for every experiment:

- `Track A`:
  fixed predictor only; no eval-time learning from validation tokens
- `Track B`:
  adaptive compression allowed, but only under strict score-before-update and single-pass rules

Record the chosen track in:
- the run command notes
- `frontier/logs/experiments.tsv`
- any future submission README

## Four conditions from issue #1017

All scored runs must satisfy all four.

1. `Strict causal dependence`
- At token `t`, prediction may depend only on the artifact and tokens before `t`
- No dependence on `x_t`, future tokens, future-token statistics, or outside side information

2. `Full normalized distribution`
- Before scoring token `t`, the system must define one real probability distribution over the full official token vocabulary
- No realized-token-only scoring with post hoc mass filling or bucket-level shortcuts

3. `Score-before-update`
- Score token `t` first
- Only after that score is fixed may model state, cache state, optimizer state, or adaptation state use token `t`

4. `Single left-to-right pass`
- Exactly one pass
- No rescoring
- No retrospective revision
- No choosing among multiple executions based on validation outcomes

## Practical experiment checks

Before accepting any result, answer these explicitly:

- Is this `Track A` or `Track B`?
- What state changes during evaluation, if any?
- Does any scored token influence its own probability?
- Is there any second pass, rescoring, or replay?
- Is the full vocabulary normalized before scoring?
- Is validation processed in the official order?

If any answer is unclear, the result is not trusted.

## BPB correctness checks

- `val_bpb` must be computed as bits per byte, not bits per token
- SentencePiece byte counting must use the actual piece table
- Handle correctly:
  - leading-space `▁`
  - byte-fallback tokens
  - boundary/control tokens with zero bytes
- Use the full validation split: `fineweb_val_*`
- Preserve validation token order

Reference implementation concept:
- build lookup tables from the tokenizer
- compute total nats, token count, and byte count
- convert with:
  `val_bpb = (total_nats / log(2)) * (token_count / byte_count)`

## Budget checks

- Artifact total must stay below `16,000,000` bytes
- Training must fit within `600s` on `8xH100 SXM`
- Evaluation must fit within an additional `600s` on the same hardware
- Any calibration or Hessian collection that consumes data to modify model state belongs to training budget, not evaluation budget

## Track-specific guidance

### Track A

Allowed:
- fixed-model sliding window
- KV-cache and inference-only speedups
- quantization and compression chosen before evaluation begins

Not allowed:
- eval-built n-gram caches that learn from validation tokens
- TTT
- LoRA adaptation on validation data
- adaptive mixtures that accumulate eval-derived statistics

### Track B

Allowed only if all four conditions still hold:
- score-first TTT
- per-document adaptation after scoring prior tokens
- causal caches built only from already-scored tokens

Not allowed:
- adapting on a token before scoring it
- rescoring a chunk after adaptation
- cache or state that reflects current or future validation tokens at scoring time

## Writeup requirements for our future submissions

Every serious candidate should be able to answer these in README form:

- What track is this submission targeting?
- What exactly changes during evaluation, if anything?
- Why does the eval path satisfy each of the four conditions?
- How is `val_bpb` computed?
- What are the train and eval wallclock times?
- What is the final artifact size?
- What seeds were run?
- What result is pre-quant, post-quant, sliding, and post-TTT if applicable?

The April 9 README is the model to follow because it explicitly documents:
- track/compliance stance
- score-first TTT behavior
- artifact size
- runtime budget
- per-seed results

## Promotion gate

Do not promote a run to "candidate" unless:

- the track is declared
- the four conditions have been checked
- BPB computation path is trusted
- runtime and artifact budgets are plausible
- the result is recorded in `frontier/logs/experiments.tsv`

## Default policy for this project

- Main lane:
  stay close to the April 9 `Track B` interpretation
- Any new eval-time adaptation:
  must be justified explicitly against the four conditions
- Any tokenizer change:
  requires explicit BPB-accounting review before the score is treated as real
