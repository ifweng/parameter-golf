# PR1787 Reproduction Lane

This lane is our first cloud benchmark target.

Target:
- upstream PR: [openai/parameter-golf#1787](https://github.com/openai/parameter-golf/pull/1787)
- upstream commit: `b667ea2576768b980dab16e1b75e9210f78a27c7`
- track: `Track B`
- expected 3-seed mean: `1.06335` post-TTT BPB
- expected artifact: about `15,939,935` bytes mean
- expected hardware: `8xH100 80GB SXM`

## Layout

- `upstream/`
  - frozen copy of the PR #1787 record folder
  - includes readable `train_gpt.py`, CaseOps prep code, tokenizer, upstream logs, and `submission.json`
- `configs/pr1787.env`
  - exact environment surface for the reproduction run
- `run_8xh100.sh`
  - cloud runner for one or more seeds
- `checks/check_pr1787_lane.py`
  - local static and lightweight compliance sanity checks
- `scripts/collect_metrics.py`
  - parses cloud logs into a summary JSON and TSV-style table

## What We Are Reproducing

PR #1787 builds on PR #1736 and adds:
- Polar Express Newton-Schulz coefficients
- `MIN_LR=0.10`
- sparse attention head-output gate
- fused softcapped CE kernel
- PR #1767 TTT settings:
  - `TTT_LORA_ALPHA=144`
  - `TTT_WARM_START_A=1`
  - `TTT_WEIGHT_DECAY=1.0`

It also keeps:
- CaseOps tokenizer/data path
- per-token original-byte sidecar for BPB
- phased score-first TTT
- Full-Hessian GPTQ + Brotli artifact path

## Local Smoke Check

Run this before spending cloud time:

```bash
.venv/bin/python frontier/lanes/pr1787/checks/check_pr1787_lane.py
```

This does not train. It verifies that the lane has the expected files, declared metrics, CaseOps roundtrip behavior, tokenizer vocab size if `sentencepiece` is available, and key compliance-relevant code paths.

## Cloud Run

The cloud runner expects a prepared CaseOps dataset. Prefer downloading the prebuilt dataset on the cloud box:

```bash
bash frontier/lanes/pr1787/bootstrap_workspace_data.sh
```

This writes `/workspace/pr1787_data.env` with the exact `CASEOPS_DATA_PATH` and `TOKENIZER_PATH` to use. The dataset directory must contain:

```text
fineweb_train_*.bin
fineweb_val_*.bin
fineweb_val_bytes_*.bin
```

Example:

```bash
CASEOPS_DATA_PATH=/workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=/workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
RUN_ROOT=/workspace/runs/pr1787 \
SEEDS="42 0 1234" \
bash frontier/lanes/pr1787/run_8xh100.sh
```

The runner writes one artifact directory per seed and then runs:

```bash
python frontier/lanes/pr1787/scripts/collect_metrics.py --run-root "$RUN_ROOT"
```

## Acceptance Criteria

Treat the reproduction as successful only if:
- train wallclock is under `600s`
- eval wallclock is under `600s`
- artifact is under `16,000,000` bytes
- post-TTT BPB is directionally near upstream `1.06335`
- CaseOps byte sidecar is used for BPB accounting
- TTT remains score-first and single-pass

## Critical Caveat

Do not use this lane as a final submission source yet. It is a benchmark lane. Any later candidate README must document our own run logs, environment, artifact bytes, and compliance evidence.
