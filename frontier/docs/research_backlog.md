# Research Backlog

This document captures the current research priorities for beating the leaderboard under the actual challenge constraints:
- artifact under `16,000,000` bytes
- train under `600s`
- eval under `600s`
- strict compliance with [issue `#1017`](https://github.com/openai/parameter-golf/issues/1017)

This is not the same as the implementation queue. It is the higher-level map of which ideas are worth investing in and why.

## April 30, 2026 Leaderboard Update

The official leaderboard has moved again. PR `#1902` merged the p-value progression update and PR `#1855` is now the accepted top public target at `1.0611` BPB.

This changes the implementation order:
- `#1855` is now the primary reproduction lane, not an audit-sensitive maybe
- `#1787` remains valuable as a readable control and regression baseline
- `#1911` / `#1738` pre-quant TTT results are not a main target because the author agreed they likely violate the stricter C3 score-before-update interpretation
- `#1948` is useful for low-risk deltas after `#1855`: LeakyReLU-square slope `0.3`, GPTQ reverse-Cholesky, and stricter GPTQ reserve accounting
- `#1984` is a promising single-seed LengthAwareTTT signal, but it should wait until our `#1855` reproduction is stable

Current immediate queue:
1. Run `frontier/lanes/pr1855/checks/check_pr1855_lane.py`.
2. Run one 1-GPU `#1855` seed as a relative benchmark and artifact/eval smoke.
3. Compare against the preserved `#1787` control only if the `#1855` lane fails or a delta regresses unexpectedly.
4. Test the `#1948` slope/GPTQ deltas on top of `#1855`.
5. Promote only clear winners to 8xH100 3-seed confirmation.

## April 30, 2026 Open Frontier Addendum

The strongest clean open PR found after the `#1855` merge is now PR `#2014`: `SP8192 CaseOps + Progressive 3k Context Growth + Short-Doc Score-First TTT`.

Reported result:
- 3-seed mean `1.05759252` post-TTT BPB
- 3-seed mean `2.31441043` val loss
- max artifact `15,984,387` bytes
- max train wallclock `596.025s`
- max validation-data TTT pass `572.4s`
- full validation coverage in all logs: `val_tokens == target_tokens == 47,853,343`

What changed versus `#1855`:
- progressive train context schedule: `1024@0.100,2048@0.700,3072@1.000`
- final train/eval/TTT context at `3072`
- `EVAL_INCLUDE_TAIL=1` so diagnostic eval and TTT eval cover the full target stream
- long-context no-Q/V TTT: `TTT_MASK=no_qv`, `TTT_Q_LORA=0`, `TTT_V_LORA=0`
- short-document score-first TTT chunks: `256:8,2000:24`
- `TTT_LOCAL_LR_MULT=0.75`
- one phased TTT pass with `PHASED_TTT_PREFIX_DOCS=2500`
- `QK_GAIN_INIT=5.25`
- AWQ-lite and asymmetric logit rescale inherited from the late-April quantization line

Updated immediate queue:
1. Make `frontier/lanes/pr2014` the main open-frontier reproduction lane.
2. Use `frontier/lanes/pr1855` as the accepted leaderboard control.
3. Run cheap 1-GPU `#2014` pre-quant and quant smoke tests before any full run.
4. Only after #2014 reproduces, test low-byte deltas: PR `#1948` reverse-Cholesky GPTQ, LeakyReLU-square slope `0.3`, stricter GPTQ reserve, short-doc TTT variants, and progressive-context schedule variants.
5. Defer PR `#1967` n-gram tilt unless its precompute path is timer-inclusive and clean under issue `#1017`.

## What This Backlog Is Based On

This backlog was refreshed on **April 28, 2026** against:
- the merged leaderboard lineage in [README.md](/Users/ifengwu/Projects/parameter-golf/README.md)
- the April 9 merged reference writeup in [README.md](/Users/ifengwu/Projects/parameter-golf/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md)
- our earlier upstream review in [upstream_frontier_review_2026-04-18.md](/Users/ifengwu/Projects/parameter-golf/frontier/docs/upstream_frontier_review_2026-04-18.md)
- authenticated GitHub connector reads from the current open PR frontier in [openai/parameter-golf](https://github.com/openai/parameter-golf)

PRs explicitly considered in this refresh:
- April-9 / earlier target stack: `#1493`, `#1518`, `#1530`, `#1532`, `#1560`
- CaseOps and clean same-family base: `#1729`, `#1736`, `#1787`
- BOS-safe SmearGate / LQER line: `#1797`, `#1851`, `#1868`, `#1855`
- non-CaseOps fallback/context: `#1790`, `#1874`
- leaderboard/audit context: `#1902`

Important change since the previous version:
- the authenticated GitHub connector is working again
- the frontier has moved materially beyond the older `#1727` / `#1732` watchlist
- the most relevant current line is now CaseOps + sparse attention gate + BOS-safe SmearGate + LQER + tuned legal TTT + improved artifact compression

## Current Frontier Snapshot

### Likely Current Best But Still Audit-Sensitive: `#1855`

`#1855` reports an SP8192 stack with CaseOps, LQER, sparse attention gate, BOS-fixed SmearGate, a 9-hyperparameter greedy tune, and per-group `lrzip`/Brotli compression.

Observed results from the PR discussion:
- 3 submitted seeds: mean `1.06108` BPB
- extra reproduction evidence discussed in `#1902`: 6-sample mean around `1.060755`
- artifacts around `15.90MB`

Why it matters:
- it is the strongest credible public signal found in this refresh
- its gain is not one idea; it is a stack of representation, architecture, TTT, quantization, hyperparameter, and compression improvements

Risks:
- extra dependency / packaging review risk from `lrzip`
- still pending leaderboard/audit handling
- hard to reproduce cleanly without first stabilizing the `#1787` / `#1851` lineage

### Strong Accepted-Looking Target Stack: `#1851` + `#1868`

`#1851` adds BOS-safe SmearGate and LQER on top of the `#1787` base, with `#1868` providing 3-seed support.

Reported 3-seed support:
- mean `1.06145` BPB
- approximate std `0.00068`
- artifact about `15.95MB`
- train about `599.5s`
- eval roughly `480s` to `526s`

Why it matters:
- this is the safest near-frontier stack to reproduce after `#1787`
- the BOS fix directly addresses the cross-document leakage issue seen in earlier SmearGate variants

### Clean Reproduction Base: `#1787`

`#1787` builds from `#1736` and reports `1.06335` BPB with:
- CaseOps tokenizer/data path
- Polar Express Newton-Schulz optimizer coefficients
- `MIN_LR=0.10`
- sparse attention gate
- fused softcapped CE kernel
- stronger legal TTT settings

Why it matters:
- it is the best practical development base before the newer SmearGate/LQER stack
- it separates enough concerns to let us reproduce and debug without starting from the most packed submission

### CaseOps Legality Signal: `#1729`, `#1736`, `#1902`

CaseOps now looks like a central, valid frontier technique rather than a side curiosity:
- `#1729` reports `1.06780` with CaseOps and tapered WD
- `#1736` reports `1.06549` with CaseOps, gated attention, quant gate, Loop45, and phased TTT
- `#1902` treats CaseOps entries as valid when the byte sidecar preserves original UTF-8 BPB accounting

Why it matters:
- representation factorization is now proven to be one of the strongest levers
- our earlier caution remains: byte accounting and reversibility must be documented and tested

### Non-CaseOps Fallback: `#1874`

`#1874` reports `1.06766` on a non-CaseOps path with Polar Express NS, `MIN_LR`, and LQER.

Why it matters:
- useful fallback if CaseOps review assumptions change
- useful control to isolate CaseOps contribution

Risks:
- inherits SmearGate-style boundary concerns from the `#1790` family unless audited with BOS/document-boundary tests

## Updated Model Of The Problem

The old framing was:
- faster training buys more optimization
- denser effective attention buys better modeling per byte

That is directionally right but incomplete. The current frontier says there are at least six independent levers:

1. **Buy more trained steps inside 600s**
- faster kernels
- fused CE
- lower overhead validation
- better GPU utilization

2. **Buy more effective modeling capacity per stored byte**
- recurrence
- parallel residuals
- sparse/gated attention outputs
- local legal token mixing such as BOS-safe SmearGate

3. **Buy a better representation of the raw text**
- CaseOps is now a major signal
- original UTF-8 byte sidecars are required for tokenizer-agnostic BPB
- reversible factorization can reduce redundant fragments before the model sees them

4. **Buy more legal adaptation during evaluation**
- score-before-update TTT remains one of the strongest Track B levers
- phased/document-boundary-aware TTT matters
- the best current line tunes TTT hyperparameters aggressively

5. **Buy a better post-quant artifact**
- LQER asymmetric correction
- clip/precision tuning
- per-row or per-group special cases for sensitive tensors

6. **Buy more artifact headroom through compression**
- byte-shuffle/Brotli was important in the April line
- the newest frontier adds tensor grouping, similarity ordering, and `lrzip`/ZPAQ-style compression
- packaging and reproducibility risk are now part of the research decision

## Must Test Now

### 1. Reproduce `#1787` As The Clean Frontier Base

Why:
- it is strong enough to matter
- it is easier to audit than `#1855`
- it introduces CaseOps, sparse attention gate, Polar Express NS, `MIN_LR`, fused CE, and stronger TTT without every later packaging complication

Pass criteria:
- artifact size and timing directionally match the PR
- CaseOps byte-sidecar BPB accounting is reproduced
- recurrence, gating, and TTT are each covered by explicit compliance checks

### 2. Add The `#1851` / `#1868` Stack

Core deltas:
- BOS-safe SmearGate
- LQER asymmetric correction
- current phased legal TTT settings

Why:
- this is the best near-frontier target that looks reproducible and reviewable
- it should become our first serious cloud confirmation candidate

Pass criteria:
- no cross-document information flow
- score-before-update behavior is tested
- post-quant/post-compression BPB improves against our `#1787` control

### 3. Test The `#1855` Hyperparameter Stack

Low-byte deltas to try after `#1851` is stable:
- `MLP_CLIP_SIGMAS=11.5`
- `EMBED_CLIP_SIGMAS=14.0`
- `WARMDOWN_FRAC=0.85`
- `BETA2=0.99`
- `TTT_BETA2=0.99`
- `TTT_WEIGHT_DECAY=0.5`
- `TTT_LORA_RANK=80`
- `SPARSE_ATTN_GATE_SCALE=0.5`
- `PHASED_TTT_PREFIX_DOCS=2500`

Why:
- this is cheap compared with architecture changes
- it is directly aligned with the strongest current public stack

### 4. Validate CaseOps End To End

Required checks:
- reversible reconstruction from CaseOps stream to original bytes
- BPB denominator uses original UTF-8 bytes
- BOS/document boundaries survive preprocessing
- validation byte sidecars match the evaluator assumption

Why:
- CaseOps is now too important to ignore
- a tiny accounting bug can invalidate an otherwise strong result

### 5. Test Compression As A First-Class Lever

Start with low-risk compression:
- byte-shuffle + Brotli compatibility
- tensor role grouping
- deterministic tensor ordering
- similarity sorting for hot rows

Then decide whether to test `lrzip`:
- only after we understand dependency, packaging, and submission reproducibility risk
- document exact install path and decompression command if used

Why:
- `#1855` suggests compression can buy enough artifact headroom to preserve useful correction terms
- this may be cheaper than finding a new model idea

## Worth Testing After The Base Is Stable

### 6. TTT-Plastic Training

Hypothesis:
- the field mostly adds TTT after training
- we may get more BPB gain per eval-second by training selected gates, scales, or LoRA surfaces to be intentionally plastic

What to test:
- late-layer-only adaptation
- gate-only adaptation
- LoRA rank/alpha surfaces trained for fast eval movement
- compare chunk-wise gain per eval-second, not only final BPB

### 7. Compression-Aware Or Quantization-Aware Training

Hypothesis:
- the challenge cares about post-quant, post-compression BPB, not raw validation loss
- a mild training penalty that improves quantization or entropy-coding behavior may pay off

What to test:
- scale-aware penalties
- clip-aware regularization
- lightweight fake-quant or quant-error proxy late in training
- preserve final BPB as the decision metric

### 8. BOS-Safe Local Mixing Beyond SmearGate

Hypothesis:
- SmearGate works because it gives a tiny, cheap previous-token mixing path
- there may be another legal local-mixing gate with better gain per byte or lower compliance risk

Constraints:
- must reset at BOS/document boundaries
- must not use future tokens
- must have a clear writeup and test

## High-Upside, High-Risk

### 9. Alternative Reversible Text Factorization

CaseOps proved that representation can matter. Other reversible factors may exist:
- punctuation spacing
- quote/dash normalization side channels
- whitespace/run-length side channels

Why risky:
- legality scrutiny will be high
- byte accounting has to be airtight
- gains can be illusory if denominator or reconstruction is wrong

Recommendation:
- keep this off the critical path until the main CaseOps line is reproduced

### 10. Small Hybrid Memory / SSM Blocks

Relevant paper:
- [Transformers are SSMs / Mamba-2](https://arxiv.org/abs/2405.21060)

Why it is tempting:
- could improve throughput and memory efficiency at the same time
- might buy more effective context modeling per parameter

Why it is risky:
- architecture pivot
- much harder to compare cleanly against the current transformer family
- likely too much engineering for the first serious cloud campaign

Recommendation:
- do not put this on the critical path
- revisit only after we have a strong same-family control and at least one credible cloud candidate

## Probably Dead Or Low Priority For Us

### 11. Multi-Token Prediction In This Regime

Relevant paper:
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)

Why it looks attractive:
- promises better sample efficiency

Why it is low priority here:
- the repo already contains negative or unstable MTP signals in the small, highly compressed regime
- the auxiliary objective costs step time and can disrupt limited model capacity

Practical stance:
- treat MTP as negative evidence unless we change the regime substantially

### 12. Full Architecture Pivots Before We Have A Stable Main Lane

Examples:
- GDN-hybrid
- Titans-style memory systems
- brand-new decoder alternatives

Why not now:
- too many degrees of freedom
- hard to attribute gains or failures
- high compliance and debugging burden

## Paper-Informed Ideas To Keep In View

These papers remain relevant, but none currently outrank reproducing the open-PR frontier:
- [FlashAttention-3](https://arxiv.org/abs/2407.08608): useful for throughput, but the challenge already strongly rewards fused/Hopper-specific kernels
- [QuaRot](https://arxiv.org/abs/2404.00456): relevant to rotation-aware quantization
- [SpinQuant](https://arxiv.org/abs/2405.16406): relevant if learned rotations can improve post-quant BPB without too much overhead
- [AWQ](https://arxiv.org/abs/2306.00978): relevant for activation/weight sensitivity and precision allocation
- [Mamba-2](https://arxiv.org/abs/2405.21060): interesting but too large a pivot before reproducing current transformer frontier
- [Multi-token Prediction](https://arxiv.org/abs/2404.19737): low priority because repo evidence is poor in this regime

## Watchlist

Immediate:
- `#1855`: likely strongest public candidate; compression/dependency risk
- `#1851` / `#1868`: strongest accepted-looking BOS-safe SmearGate + LQER stack
- `#1787`: clean base to reproduce first
- `#1902`: leaderboard and BOS/CaseOps audit context

Secondary:
- `#1797`: useful corrected SmearGate/LQER history; original result had leakage bias
- `#1874`: non-CaseOps fallback and LQER context
- `#1729` / `#1736`: CaseOps legality and early implementation context

Do not build on without extra scrutiny:
- old SmearGate variants without BOS masking
- PPM-D / byte-mixture lines pending C2 or rule interpretation
- submissions whose gain depends on unclear byte accounting

## Recommended Research Order

1. Rebase the project target from April-9 merged SOTA to the `#1787` -> `#1851/#1868` -> `#1855` lineage.
2. Reproduce `#1787` on cloud as the clean frontier base.
3. Validate CaseOps byte accounting and BOS/document-boundary handling.
4. Add BOS-safe SmearGate and LQER from the `#1851/#1868` line.
5. Apply the `#1855` low-risk hyperparameter stack.
6. Test per-group compression only after the model path is stable.
7. Only then spend time on novel papers or architecture pivots.

## Default Kill Criteria

Kill an idea early if:
- it is not clearly compliant with issue `#1017`
- it improves pre-quant loss but worsens final post-quant BPB
- it costs too much train or eval wallclock for the gain
- it cannot be explained clearly enough for a leaderboard README
- it requires a dependency or packaging step that cannot be reproduced cleanly
