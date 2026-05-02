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

### Candidate 0: PR #1953 Clean LongCtx NoQV Baseline

Purpose:
- reproduce the strongest clean, compact baseline we currently trust before adding novelty
- use it as the control for no-artifact or eval-only improvements

Lane:
- `frontier/lanes/pr1953/README.md`

Cheap smoke:

```bash
source /workspace/pr1953_data.env
NPROC_PER_NODE=1 \
SEEDS="42" \
MAX_WALLCLOCK_SECONDS=120 \
PREQUANT_ONLY=1 \
COMPRESSOR=brotli \
RUN_ROOT=/workspace/runs/pr1953_prequant_smoke \
bash frontier/lanes/pr1953/run_8xh100.sh
```

Command:

```bash
source /workspace/pr1953_data.env
NPROC_PER_NODE=8 \
SEEDS="42 0 1234" \
RUN_ROOT=/workspace/runs/pr1953_3seed \
bash frontier/lanes/pr1953/run_8xh100.sh
```

Expected envelope:
- track: `Track B`
- upstream score: `1.05855370` post-TTT BPB, 3-seed mean
- upstream loss: `2.31650787` nats, 3-seed mean
- upstream artifact: about `15,990,177` bytes mean, max `15,992,914`
- train time: about `596.1s`
- eval time: about `430s` to `513s`
- promotion gate: reproduce seed 42 directionally before opening the entropy-adaptive tilt experiment

Critical prerequisite:
- a prepared CaseOps dataset with `fineweb_val_bytes_*.bin` sidecars
- `lrzip` installed on the cloud image if `COMPRESSOR=pergroup`

### Candidate 1: PR1953 Exp01 Weighted TTT

Purpose:
- improve the main observed gap: local TTT recovery
- keep model architecture, training recipe, quantization, and artifact size unchanged

Lane:
- `frontier/lanes/pr1953_exp01_weighted_ttt/`

Mechanism:
- after score accumulation, compute detached per-doc chunk losses
- weight local LoRA TTT updates continuously by relative chunk difficulty
- default: `alpha=0.5`, `min=0.5`, `max=1.75`, `ema=0.98`

Cheapest eval-only command if a PR1953 seed-42 artifact exists:

```bash
source /workspace/pr1953_data.env
TTT_EVAL_ONLY=1 \
ARTIFACT_ROOT=/workspace/runs/pr1953_seed42 \
RUN_ROOT=/workspace/runs/pr1953_exp01_weighted_evalonly_seed42 \
NPROC_PER_NODE=8 \
SEEDS="42" \
bash frontier/lanes/pr1953_exp01_weighted_ttt/run_8xh100.sh
```

Full seed-42 command:

```bash
source /workspace/pr1953_data.env
NPROC_PER_NODE=8 \
SEEDS="42" \
RUN_ROOT=/workspace/runs/pr1953_exp01_weighted_seed42 \
bash frontier/lanes/pr1953_exp01_weighted_ttt/run_8xh100.sh
```

Promotion gate:
- seed 42 must beat reproduced PR #1953 seed 42 post-TTT BPB
- logs must show `ttt_loss_weighted:` and `lw:`
- eval must remain under `600s`
- artifact must remain under `16,000,000` bytes

### Candidate 2: Novel PR1953 Entropy-Adaptive Token Tilt

Purpose:
- add a no-artifact, strictly causal score-time correction on top of PR #1953
- improve over fixed token-only n-gram tilt by adapting boost strength to prefix-only confidence and model uncertainty

Lane:
- planned: `frontier/lanes/pr1953_exp01_entropy_tilt/`

Promotion gate:
- only open after Candidate 1 is tested
- no within-word, word-start, agreement, or target-token-gated channels
- n-gram work must be inside the eval timer
- eval must remain under `600s`
- artifact must remain under `16,000,000` bytes

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

### Deferred: PR1855 Exp02 #2060 Retune On #2101

Purpose:
- preserve the prior queue for forensic comparison only
- do not spend fresh cloud budget here before PR #1953 reproduces

Lane:
- start from `frontier/lanes/pr1855_exp01_awqlite_asymlogit/README.md`

Target deltas:
- config-only retune from PR #2060: `MATRIX_LR=0.028`, `LQER_RANK=2`, `LQER_ASYM_GROUP=32`, `LQER_TOP_K=4`, `TTT_LOCAL_LR_MULT=0.80`
- keep #2101 TTT phasing for the first run so the A/B isolates these five knobs

Full seed-42 A/B:

```bash
source /workspace/pr1855_exp01_data.env
CONFIG_PATH=frontier/lanes/pr1855_exp01_awqlite_asymlogit/configs/exp02_2060_retune.env \
NPROC_PER_NODE=8 \
SEEDS="42" \
RUN_ROOT=/workspace/runs/pr1855_exp02_2060retune_seed42 \
bash frontier/lanes/pr1855_exp01_awqlite_asymlogit/run_8xh100.sh
```

Cheapest TTT-only probe if Candidate 0 already produced an artifact:

```bash
source /workspace/pr1855_exp01_data.env
CONFIG_PATH=frontier/lanes/pr1855_exp01_awqlite_asymlogit/configs/exp02_tttlocal080_only.env \
ARTIFACT_ROOT=/workspace/runs/pr1855_exp01_seed42 \
RUN_ROOT=/workspace/runs/pr1855_exp02_tttlocal080_seed42 \
TTT_EVAL_ONLY=1 \
NPROC_PER_NODE=8 \
SEEDS="42" \
bash frontier/lanes/pr1855_exp01_awqlite_asymlogit/run_8xh100.sh
```

Expected envelope:
- target score: must beat the old #2101 reproduction on post-TTT BPB by enough to justify three-seed confirmation
- artifact: should remain below `16,000,000` bytes
- risk: low to medium because the artifact margin is only about `21KB` in the reference run

### Stopped: Novel Loss-Gated TTT On #2101

Purpose:
- keep the prior novel eval-time adaptation attempt documented as negative evidence
- reduce harmful LoRA updates on easy chunks by updating only from already-scored high-loss chunks

Reason:
- observed average score was `1.06076055`, worse than PR #1953 and not worth promoting

Lane:
- `frontier/lanes/pr1855_exp01_awqlite_asymlogit/README.md`

Mechanism:
- after `_accumulate_bpb(...)`, compute per-document chunk loss
- keep documents above `mean + z * std` among active documents, with a minimum kept fraction
- default experiment: `TTT_LOSS_GATE_Z=-0.25`, `TTT_LOSS_GATE_MIN_FRAC=0.35`, `TTT_LOSS_GATE_WARMUP_CHUNKS=1`

Cheapest eval-only command if a #2101 artifact exists:

```bash
source /workspace/pr1855_exp01_data.env
CONFIG_PATH=frontier/lanes/pr1855_exp01_awqlite_asymlogit/configs/exp03_lossgated_ttt.env \
ARTIFACT_ROOT=/workspace/runs/pr1855_exp01_seed42 \
RUN_ROOT=/workspace/runs/pr1855_exp03_lossgated_ttt_seed42 \
TTT_EVAL_ONLY=1 \
NPROC_PER_NODE=8 \
SEEDS="42" \
bash frontier/lanes/pr1855_exp01_awqlite_asymlogit/run_8xh100.sh
```

Promotion gate:
- positive means post-TTT BPB improves, not just eval time
- eval time must remain under `600s`
- logs should show `lg:kept/total` so the gate is actually active
- if positive, stack with Exp02 using `configs/exp03_2060_lossgated_ttt.env`

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
