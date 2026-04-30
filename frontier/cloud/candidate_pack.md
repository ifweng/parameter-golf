# Cloud Candidate Pack

This file is the shortlist for 8xH100 runs. Keep it narrow.

## Promotion Rules

Promote a candidate only if all of the following hold locally or on cheap smoke checks:
- the change is internally consistent and reproducible
- the expected artifact stays below `16,000,000` bytes
- the eval path remains legal for the intended track
- the candidate beats the current internal best on the metric it is supposed to improve

Compliance reference:
- all candidate promotion checks must also satisfy `frontier/docs/compliance_checklist.md`

## Current target hierarchy

### Candidate 0: PR #1855 accepted frontier reproduction

Purpose:
- reproduce the current accepted leaderboard stack before testing our own deltas
- make the 1-GPU cloud setup produce durable metrics and artifacts

Lane:
- `frontier/lanes/pr1855/README.md`

Command:

```bash
source /workspace/pr1855_data.env
NPROC_PER_NODE=8 \
SEEDS="42 0 1234" \
RUN_ROOT=/workspace/runs/pr1855_3seed \
bash frontier/lanes/pr1855/run_8xh100.sh
```

Expected envelope:
- track: `Track B`
- upstream score: `1.06107587` post-TTT BPB, 3-seed mean
- upstream artifact: about `15,901,919` bytes mean, max `15,907,550`
- train time: about `599.5s`
- eval time: about `455s` to `509s`
- promotion gate: reproduce directionally with trusted CaseOps byte sidecar accounting, BOS-safe SmearGate, and all budgets intact

Critical prerequisite:
- a prepared CaseOps dataset with `fineweb_val_bytes_*.bin` sidecars
- `lrzip` installed on the cloud image if `COMPRESSOR=pergroup`

### Control: PR #1787 clean frontier reproduction

Purpose:
- preserve a readable previous-frontier control for regressions and ablations

Lane:
- `frontier/lanes/pr1787/README.md`

Expected envelope:
- upstream score: `1.06335` post-TTT BPB, 3-seed mean
- upstream artifact: about `15,939,935` bytes mean
- train time: about `599.5s`
- eval time: about `416s` to `526s`
- promotion gate: use only as a regression/control lane now that #1855 is official top

### Candidate 1: PR #1948 low-risk deltas on top of PR #1855

Purpose:
- test cheap changes that should not disturb compliance or model shape

Target deltas:
- LeakyReLU-square negative slope `0.3`
- GPTQ reverse-Cholesky speed/compliance implementation
- stricter GPTQ reserve accounting, likely `8s` to `16s` in final confirmation

Expected envelope:
- target score: must beat our internal #1855 reproduction on post-TTT BPB
- artifact: should remain below `16,000,000` bytes
- risk: low for slope, medium for GPTQ path changes because serialization/eval timing must be rechecked

### Candidate 2: PR #1984 LengthAwareTTT-style scheduling

Purpose:
- investigate the strongest recent single-seed signal only after the accepted #1855 lane is stable

Target deltas:
- length-aware/phased TTT scheduling changes
- keep score-before-update and single-pass constraints explicit

Expected envelope:
- public evidence: single seed around `1.06018`, not a valid 3-seed record
- risk: medium until reproduced cleanly
- promotion gate: one seed must clearly improve before spending on 3-seed confirmation

## Deferred watchlist

- PR #1911 / PR #1738 pre-quant TTT line: defer as likely invalid under stricter C3 score-before-update interpretation
- MHA/KV=8 path from PR #1987: defer because it trails PR #1855 and tightens eval time
- Mamba/SSM branches: interesting research, not competitive with the current transformer frontier yet
- PPM-D / byte-mixture lines: defer until the C2 interpretation is settled
