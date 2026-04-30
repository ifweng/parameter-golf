# Cloud Candidate Pack

This file is the shortlist for 8xH100 runs. Keep it narrow.

## Promotion Rules

Promote a candidate only if all of the following hold locally or on cheap smoke checks:
- the change is internally consistent and reproducible
- the expected artifact stays below `16,000,000` bytes
- the eval path remains legal for the intended track
- the candidate beats the current internal best on the metric it is supposed to improve

Compliance reference:
- all candidate promotion checks must also satisfy [compliance_checklist.md](/Users/ifengwu/Projects/parameter-golf/frontier/docs/compliance_checklist.md)

## Current target hierarchy

### Candidate 0: PR #1787 clean frontier reproduction

Purpose:
- establish the first real 8xH100 benchmark lane for the current frontier
- reproduce a strong but still readable stack before adding newer SmearGate/LQER changes

Lane:
- [PR1787 Reproduction Lane](/Users/ifengwu/Projects/parameter-golf/frontier/lanes/pr1787/README.md)

Command:

```bash
CASEOPS_DATA_PATH=/workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
RUN_ROOT=/workspace/runs/pr1787 \
SEEDS="42 0 1234" \
bash frontier/lanes/pr1787/run_8xh100.sh
```

Expected envelope:
- track: `Track B`
- upstream score: `1.06335` post-TTT BPB, 3-seed mean
- upstream artifact: about `15,939,935` bytes mean
- train time: about `599.5s`
- eval time: about `416s` to `526s`
- promotion gate: reproduce directionally with trusted CaseOps byte sidecar accounting and all budgets intact

Critical prerequisite:
- a prepared CaseOps dataset with `fineweb_val_bytes_*.bin` sidecars

### Candidate 1: PR #1851/#1868 BOS-safe SmearGate + LQER stack

Purpose:
- first attempt to match the strongest accepted-looking post-#1787 stack

Target deltas after Candidate 0 works:
- BOS-safe SmearGate
- LQER asymmetric rank-4 quant correction
- same CaseOps and phased score-first TTT path

Expected envelope:
- upstream score: `1.06145` post-TTT BPB, 3-seed mean
- artifact: about `15.95 MB`
- train time: about `599.5s`
- eval time: about `480s` to `526s`
- promotion gate: improve over our PR #1787 reproduction without any BOS/document-boundary leak

### Candidate 2: PR #1855 hparam stack without risky compression first

Purpose:
- test the strongest public hparam deltas while avoiding the `lrzip` packaging question initially

Target deltas:
- `MLP_CLIP_SIGMAS=11.5`
- `EMBED_CLIP_SIGMAS=14.0`
- `WARMDOWN_FRAC=0.85`
- `BETA2=0.99`
- `TTT_BETA2=0.99`
- `TTT_WEIGHT_DECAY=0.5`
- `TTT_LORA_RANK=80`
- `SPARSE_ATTN_GATE_SCALE=0.5`
- `PHASED_TTT_PREFIX_DOCS=2500`

Expected envelope:
- target score: below our Candidate 1 reproduction
- artifact: may be slightly larger than #1855 if Brotli-only compression is used
- promotion gate: score improves enough to justify testing per-group compression

### Candidate 3: PR #1855 per-group compression

Purpose:
- recover artifact headroom after the score path is stable

Target deltas:
- tensor role grouping
- similarity row ordering for hot tensors
- per-group compression, possibly `lrzip`/ZPAQ if reproducibility risk is acceptable

Expected envelope:
- upstream score evidence: `1.06108` 3-seed mean, with later 6-sample discussion around `1.060755`
- artifact: about `15.90 MB`
- risk: dependency and packaging audit
- promotion gate: exact decompression path is reproducible and clearly documentable

## Deferred watchlist

- non-CaseOps fallback from `#1874` if CaseOps assumptions change
- Mamba/SSM branches:
  interesting but not competitive with the current transformer frontier
- PPM-D / byte-mixture lines:
  defer until the C2 interpretation is settled
