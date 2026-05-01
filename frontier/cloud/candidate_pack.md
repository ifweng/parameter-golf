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

### Candidate 0: PR #1855 Exp01 AWQ-lite + AsymLogit

Purpose:
- improve directly from the accepted PR #1855 stack without adopting the PR #2018 n-gram/gated-XSA base
- reproduce the lowest-risk PR1855-derived upstream improvement from PR #2101 before testing our own deltas

Lane:
- `frontier/lanes/pr1855_exp01_awqlite_asymlogit/README.md`

Cheap smoke:

```bash
source /workspace/pr1855_exp01_data.env
NPROC_PER_NODE=1 \
SEEDS="42" \
MAX_WALLCLOCK_SECONDS=120 \
PREQUANT_ONLY=1 \
COMPRESSOR=brotli \
RUN_ROOT=/workspace/runs/pr1855_exp01_prequant_smoke \
bash frontier/lanes/pr1855_exp01_awqlite_asymlogit/run_8xh100.sh
```

Command:

```bash
source /workspace/pr1855_exp01_data.env
NPROC_PER_NODE=8 \
SEEDS="42 0 1234" \
RUN_ROOT=/workspace/runs/pr1855_exp01_3seed \
bash frontier/lanes/pr1855_exp01_awqlite_asymlogit/run_8xh100.sh
```

Expected envelope:
- track: `Track B`
- upstream score: `1.05845438` post-TTT BPB, 3-seed mean
- upstream loss: `2.31721592` nats, 3-seed mean
- upstream artifact: about `15,978,063` bytes mean, max `15,978,610`
- train time: about `592.1s` plus GPTQ Hessian collection within the 600s data-access budget
- eval time: about `508s` to `520s`
- promotion gate: beat reproduced PR #1855 on post-TTT BPB while preserving trusted CaseOps byte sidecar accounting, full validation target coverage, BOS-safe SmearGate, score-first TTT, and all budgets intact

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

### Candidate 1: Small PR1855 Exp01 follow-up A/Bs

Purpose:
- test only cheap same-family knobs after Candidate 0 reproduces cleanly

Lane:
- start from `frontier/lanes/pr1855_exp01_awqlite_asymlogit/README.md`

Target deltas:
- config-only retune from PR #2060: `MATRIX_LR=0.028`, `LQER_RANK=2`, `LQER_ASYM_GROUP=32`, `LQER_TOP_K=4`, `TTT_LOCAL_LR_MULT=0.80`
- optional `GRAD_CENTRALIZE=1` single-seed A/B only after the reproduction is stable
- optional tiny `LABEL_SMOOTH` A/B only if gradient centralization is neutral or positive

Cheap comparison commands:

```bash
source /workspace/pr1855_exp01_data.env
NPROC_PER_NODE=8 \
SEEDS="42" \
MATRIX_LR=0.028 \
LQER_RANK=2 \
LQER_ASYM_GROUP=32 \
LQER_TOP_K=4 \
TTT_LOCAL_LR_MULT=0.80 \
RUN_ROOT=/workspace/runs/pr1855_exp01_lqer_g32_top4_seed42 \
bash frontier/lanes/pr1855_exp01_awqlite_asymlogit/run_8xh100.sh
```

Expected envelope:
- target score: must beat Candidate 0 on post-TTT BPB by enough to justify three-seed confirmation
- artifact: should remain below `16,000,000` bytes
- risk: low to medium because the artifact margin is only about `21KB` in the reference run

### Stopped: PR #2014 / reverse-Cholesky / slope 0.3 lane

Purpose:
- keep the prior work documented, but do not spend more cloud budget unless we need a forensic comparison

Reason:
- our local/cloud exp01 attempts produced worse BPB and tighter size/eval behavior than the PR1855-derived lane
- reverse-Cholesky and slope `0.3` are no longer the least-effort path to improvement

Lane:
- `frontier/lanes/pr2014_exp01_slope03_rchgptq/README.md`

## Deferred watchlist

- PR #1911 / PR #1738 pre-quant TTT line: defer as likely invalid under stricter C3 score-before-update interpretation
- PR #2018 / PR #1967 n-gram tilt line: user explicitly does not want this as the base; defer unless we open a separate high-risk lane
- MHA/KV=8 path from PR #1987: defer because it trails PR #1855 and tightens eval time
- Mamba/SSM branches: interesting research, not competitive with the current transformer frontier yet
- PPM-D / byte-mixture lines: defer until the C2 interpretation is settled
