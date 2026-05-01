# PR1855 Exp01: AWQ-lite + AsymLogit

This lane keeps PR #1855 as the base and mirrors upstream PR #2101 as the first low-risk improvement attempt.

Target:
- base: [openai/parameter-golf#1855](https://github.com/openai/parameter-golf/pull/1855)
- improvement reference: [openai/parameter-golf#2101](https://github.com/openai/parameter-golf/pull/2101)
- upstream PR #2101 head: `c71bff83f225b64c536c7acdc797430e4e6f2887`
- expected 3-seed mean: `1.05845` post-TTT BPB
- expected artifact: about `15,978,063` bytes mean, max `15,978,610`
- expected eval: about `508s` to `520s` on 8xH100 SXM

## Why This Lane

We are intentionally not using the PR #2018 n-gram/gated-XSA stack as the base. This lane improves directly from the accepted PR #1855 family and keeps the compliance profile simpler:
- no n-gram or byte-mixer path
- CaseOps byte sidecar remains enabled for BPB accounting
- score-first phased TTT is inherited
- added features are quantization/logit-shape changes, not validation-data memorization machinery

## Added Deltas Versus PR1855

- `AWQ_LITE_ENABLED=1`
- `AWQ_LITE_GROUP_TOP_K=1`
- `ASYM_LOGIT_RESCALE=1`
- `TTT_MASK=no_qv`
- `TTT_LOCAL_LR_MULT=0.75`
- `GPTQ_RESERVE_SECONDS=8.0`
- optional code features from PR #2101 are present but off: `GRAD_CENTRALIZE=0`, `LABEL_SMOOTH=0.0`

## Static Check

```bash
python3 frontier/lanes/pr1855_exp01_awqlite_asymlogit/checks/check_pr1855_exp01_lane.py
```

## Cloud Data Bootstrap

This lane uses the same CaseOps SP8192 dataset as PR1855. On the cloud box:

```bash
bash frontier/lanes/pr1855_exp01_awqlite_asymlogit/bootstrap_workspace_data.sh
```

Install the per-group compressor dependency if using submission-style compression:

```bash
sudo apt-get update
sudo apt-get install -y lrzip
```

## Cheap Smoke

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

## 8xH100 Seed 42 Confirmation

```bash
source /workspace/pr1855_exp01_data.env
NPROC_PER_NODE=8 \
SEEDS="42" \
RUN_ROOT=/workspace/runs/pr1855_exp01_seed42 \
bash frontier/lanes/pr1855_exp01_awqlite_asymlogit/run_8xh100.sh
```

## 8xH100 Three-Seed Confirmation

```bash
source /workspace/pr1855_exp01_data.env
NPROC_PER_NODE=8 \
SEEDS="42 0 1234" \
RUN_ROOT=/workspace/runs/pr1855_exp01_3seed \
bash frontier/lanes/pr1855_exp01_awqlite_asymlogit/run_8xh100.sh
```

## Exp02: Improve Beyond #2101

The first improvement attempt is a #2060-style five-knob retune on top of the #2101 source:

```bash
source /workspace/pr1855_exp01_data.env
CONFIG_PATH=frontier/lanes/pr1855_exp01_awqlite_asymlogit/configs/exp02_2060_retune.env \
NPROC_PER_NODE=8 \
SEEDS="42" \
RUN_ROOT=/workspace/runs/pr1855_exp02_2060retune_seed42 \
bash frontier/lanes/pr1855_exp01_awqlite_asymlogit/run_8xh100.sh
```

This forces:
- `MATRIX_LR=0.028`
- `LQER_RANK=2`
- `LQER_ASYM_GROUP=32`
- `LQER_TOP_K=4`
- `TTT_LOCAL_LR_MULT=0.80`

If a #2101 artifact already exists, test the cheapest TTT-only component first:

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

## Promotion Criteria

Promote only if:
- post-TTT BPB beats reproduced PR1855, not just diagnostic quantized BPB
- max artifact stays below `16,000,000` bytes
- final TTT eval stays below `600s`
- logs show `CASEOPS_ENABLED=1` and validation byte sidecars are present
- `GRAD_CENTRALIZE` and `LABEL_SMOOTH` remain off unless explicitly tested as separate A/B experiments
