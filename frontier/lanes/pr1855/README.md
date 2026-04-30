# PR1855 Frontier Lane

This lane is the current accepted leaderboard target.

Target:
- upstream PR: [openai/parameter-golf#1855](https://github.com/openai/parameter-golf/pull/1855)
- upstream head commit: `1e439663209730edeac34e659039d7de62d85908`
- merged via: `510d03e0fc355406c9fd06f92d23b8c5aedea7fb`
- track: `Track B`
- expected 3-seed mean: `1.06107587` post-TTT BPB
- expected artifact: about `15,901,919` bytes mean, max `15,907,550`
- expected hardware: `8xH100 80GB SXM`

## Layout

- `upstream/`
  - frozen copy of the PR #1855 record folder
  - includes readable `train_gpt.py`, CaseOps prep code, tokenizer, upstream logs, `requirements.txt`, and `submission.json`
- `configs/pr1855.env`
  - environment surface for reproducing the accepted stack
- `run_8xh100.sh`
  - cloud runner for one or more seeds; also accepts `NPROC_PER_NODE=1` for budget smoke runs
- `bootstrap_workspace_data.sh`
  - downloads the shared SP8192 CaseOps dataset to `/workspace`
- `checks/check_pr1855_lane.py`
  - local static and lightweight compliance sanity checks
- `scripts/collect_metrics.py`
  - wrapper around the shared metrics parser

## What We Are Reproducing

PR #1855 builds on the PR #1787 line and adds:
- LQER asymmetric int4 rank-4 quant-error correction on the top 3 tensors
- SparseAttnGate with `SPARSE_ATTN_GATE_SCALE=0.5`
- SmearGate with the BOS cross-document leak fix
- per-group `lrzip` + Brotli compression
- a 9-hyperparameter stack:
  - `MLP_CLIP_SIGMAS=11.5`
  - `EMBED_CLIP_SIGMAS=14.0`
  - `WARMDOWN_FRAC=0.85`
  - `BETA2=0.99`
  - `TTT_BETA2=0.99`
  - `TTT_WEIGHT_DECAY=0.5`
  - `TTT_LORA_RANK=80`
  - `SPARSE_ATTN_GATE_SCALE=0.5`
  - `PHASED_TTT_PREFIX_DOCS=2500`

The public PR discussion accepted the `lrzip` system dependency when it is declared as setup prerequisite and used only as an already-installed local binary during serialization/evaluation.

## Local Static Check

Run this before spending cloud time:

```bash
python3 frontier/lanes/pr1855/checks/check_pr1855_lane.py
```

This does not train. It verifies provenance, published metrics, CaseOps roundtrip behavior, tokenizer vocab size if `sentencepiece` is available, BOS-safe SmearGate, LQER, per-group compression, and BPB byte-sidecar paths.

## Cloud Data Bootstrap

On the cloud box:

```bash
bash frontier/lanes/pr1855/bootstrap_workspace_data.sh
```

This writes `/workspace/pr1855_data.env` with the exact `CASEOPS_DATA_PATH`, `TOKENIZER_PATH`, and default `RUN_ROOT`.

PR #1855 also requires the `lrzip` system binary when `COMPRESSOR=pergroup`:

```bash
sudo apt-get update
sudo apt-get install -y lrzip
```

## 1-GPU Benchmark Command

Use this for budget-limited relative benchmarking. It is not leaderboard-equivalent.

```bash
source /workspace/pr1855_data.env
NPROC_PER_NODE=1 \
SEEDS="42" \
RUN_ROOT=/workspace/runs/pr1855_seed42_1gpu \
bash frontier/lanes/pr1855/run_8xh100.sh
```

## 8xH100 Confirmation Command

Use this only for narrow confirmation once 1-GPU smoke runs and compliance checks pass:

```bash
source /workspace/pr1855_data.env
NPROC_PER_NODE=8 \
SEEDS="42 0 1234" \
RUN_ROOT=/workspace/runs/pr1855_3seed \
bash frontier/lanes/pr1855/run_8xh100.sh
```

The runner writes one artifact directory per seed and then runs:

```bash
python3 frontier/lanes/pr1855/scripts/collect_metrics.py --run-root "$RUN_ROOT"
```

The metrics parser writes both:
- `metrics_summary.json`
- `metrics_summary.md`

## Acceptance Criteria

Treat reproduction as successful only if:
- train wallclock is under `600s`
- TTT eval wallclock is under `600s`
- artifact is under `16,000,000` bytes
- post-TTT BPB is directionally near upstream `1.06108`
- CaseOps byte sidecar is used for BPB accounting
- SmearGate masks BOS positions in both train and TTT paths
- no pre-score validation-token optimization is introduced

## Next Delta Queue

After this reproduces cleanly, the least-effort deltas to test are:
- LeakyReLU-square slope `0.3` from PR #1948
- GPTQ reverse-Cholesky speed/compliance path from PR #1948
- stricter GPTQ reserve accounting, likely `8s` to `16s` for final runs
- LengthAwareTTT-style scheduling only after the fixed `#1855` path is stable

