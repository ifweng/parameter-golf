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

### Candidate 0: PR #2014 open frontier reproduction

Purpose:
- reproduce the strongest clean open frontier stack before testing our own deltas
- make the 1-GPU cloud setup produce durable metrics and artifacts on the current target

Lane:
- `frontier/lanes/pr2014/README.md`

Command:

```bash
source /workspace/pr2014_data.env
NPROC_PER_NODE=8 \
SEEDS="42 314 0" \
RUN_ROOT=/workspace/runs/pr2014_3seed \
bash frontier/lanes/pr2014/run_8xh100.sh
```

Expected envelope:
- track: `Track B`
- upstream score: `1.05759252` post-TTT BPB, 3-seed mean
- upstream loss: `2.31441043` nats, 3-seed mean
- upstream artifact: about `15,982,485` bytes mean, max `15,984,387`
- train time: about `596.0s`
- eval time: about `490s` to `572s`
- promotion gate: reproduce directionally with trusted CaseOps byte sidecar accounting, full validation target coverage, BOS-safe SmearGate, score-first TTT, and all budgets intact

Critical prerequisite:
- a prepared CaseOps dataset with `fineweb_val_bytes_*.bin` sidecars
- `lrzip` installed on the cloud image if `COMPRESSOR=pergroup`

### Control: PR #1855 accepted frontier reproduction

Purpose:
- preserve the current accepted leaderboard stack as a compliance and regression control

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
- upstream score: `1.06107587` post-TTT BPB, 3-seed mean
- upstream artifact: about `15,901,919` bytes mean, max `15,907,550`
- train time: about `599.5s`
- eval time: about `455s` to `509s`
- promotion gate: use as the accepted-control lane when a #2014 delta is ambiguous

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
- promotion gate: use only as a regression/control lane now that #1855 is official top and #2014 is the open frontier

### Candidate 1: PR #1948 low-risk deltas on top of PR #2014

Purpose:
- test cheap changes that should not disturb compliance or model shape

Target deltas:
- LeakyReLU-square negative slope `0.3`
- GPTQ reverse-Cholesky speed/compliance implementation
- stricter GPTQ reserve accounting, likely `8s` to `12s` in final confirmation

Expected envelope:
- target score: must beat our internal #2014 reproduction on post-TTT BPB, or preserve BPB while materially improving timing/compliance margin
- artifact: should remain below `16,000,000` bytes
- risk: low for slope, medium for GPTQ path changes because serialization/eval timing must be rechecked

### Candidate 2: PR #2014 schedule search

Purpose:
- investigate the highest-probability same-family BPB improvements after the #2014 lane is stable

Target deltas:
- short-doc TTT schedule variants such as `128:8,512:16,2000:24`
- progressive context schedule variants that move the 2k to 3k transition earlier or later
- keep score-before-update, full validation-tail coverage, and single-pass constraints explicit

Expected envelope:
- public evidence: #2014 3-seed mean `1.05759252`
- risk: medium because the 8xH100 timing margin is already tight
- promotion gate: one seed must clearly improve the #2014 control before spending on 3-seed confirmation

## Deferred watchlist

- PR #1911 / PR #1738 pre-quant TTT line: defer as likely invalid under stricter C3 score-before-update interpretation
- PR #1967 n-gram tilt line: defer unless the precompute is timer-inclusive and remains under budget
- MHA/KV=8 path from PR #1987: defer because it trails PR #1855 and tightens eval time
- Mamba/SSM branches: interesting research, not competitive with the current transformer frontier yet
- PPM-D / byte-mixture lines: defer until the C2 interpretation is settled
