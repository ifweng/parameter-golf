# PR #1953 Frontier Lane

This lane is a frozen reproduction target for PR #1953:

- upstream PR: <https://github.com/openai/parameter-golf/pull/1953>
- reported 3-seed BPB: `1.05855370`
- reported 3-seed val loss: `2.31650787`
- reported max artifact: `15,992,914` bytes
- reported max eval: `513.1s`
- audit status: clean in issue #2127 because it uses the canonical HF CaseOps snapshot, not local `prepare_caseops_data.py`

It is our new clean baseline for low-artifact or eval-only improvements. Do not edit files under `upstream/`; create a sibling experiment lane for modifications.

## Stack Summary

PR #1953 starts from PR #1945, itself PR #1855 plus AWQ-lite and AsymLogit Rescale, then adds:

- `EVAL_SEQ_LEN=2560`
- `TTT_EVAL_SEQ_LEN=2560`
- `TTT_MASK=no_qv`, `TTT_Q_LORA=0`, `TTT_V_LORA=0`
- `TTT_LOCAL_LR_MULT=0.75`
- `QK_GAIN_INIT=5.25`

The tight constraint is artifact size: the upstream max is only about `7KB` below the `16,000,000` byte cap. This makes PR #1953 a poor base for adding stored parameters, but a good base for eval-only methods or source-only scoring changes.

## Local Check

```bash
python3 frontier/lanes/pr1953/checks/check_pr1953_lane.py
```

This does not train. It verifies provenance, published metrics, canonical CaseOps log evidence, config values, and static compliance-relevant code paths.

## Cloud Data

On RunPod or another cloud machine:

```bash
cd /workspace/parameter-golf
bash frontier/lanes/pr1953/bootstrap_workspace_data.sh
source /workspace/pr1953_data.env
```

Install `lrzip` before submission-grade `COMPRESSOR=pergroup` runs:

```bash
sudo apt-get update
sudo apt-get install -y lrzip
```

## Cheap Smoke

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

## Seed 42 Reproduction

```bash
source /workspace/pr1953_data.env
NPROC_PER_NODE=8 \
SEEDS="42" \
RUN_ROOT=/workspace/runs/pr1953_seed42 \
bash frontier/lanes/pr1953/run_8xh100.sh
```

## Three-Seed Reproduction

```bash
source /workspace/pr1953_data.env
NPROC_PER_NODE=8 \
SEEDS="42 0 1234" \
RUN_ROOT=/workspace/runs/pr1953_3seed \
bash frontier/lanes/pr1953/run_8xh100.sh
```

Metrics are written to:

- `$RUN_ROOT/metrics_summary.json`
- `$RUN_ROOT/metrics_summary.md`

## First Improvement Direction

The first experiment should be a sibling lane, not an edit here:

- `frontier/lanes/pr1953_exp01_entropy_tilt/`

Target idea:

- strictly causal token-only n-gram hinting
- no within-word, word-start, agreement, or target-token-gated channels
- full-vocab renormalization
- adaptive boost based on prefix-only n-gram confidence and model uncertainty
- precompute/scoring inside eval timer

Promotion gate:

- beat PR #1953 seed-42 post-TTT BPB before any three-seed run
- preserve `train_shards: 80`, validation byte sidecars, score-before-update TTT, eval under `600s`, and artifact under `16,000,000` bytes
