# PR2014 Exp01: Slope 0.3 + Reverse-Cholesky GPTQ

This is the first controlled improvement lane on top of frozen PR #2014.

Purpose:
- keep `frontier/lanes/pr2014` as the clean reproduction baseline
- test two low-byte PR #1948 deltas on the #2014 stack
- preserve the #2014 data, tokenizer, progressive context schedule, TTT schedule, quantization settings, and compression defaults

Delta surface:
- `LEAKY_RELU_SQ_SLOPE=0.3`
- `GPTQ_HINV_MODE=reverse_cholesky`

Everything else is inherited from:
- [PR2014 lane](/Users/ifengwu/.codex/worktrees/b8bb/parameter-golf/frontier/lanes/pr2014/README.md)
- [exp01 config](/Users/ifengwu/.codex/worktrees/b8bb/parameter-golf/frontier/lanes/pr2014_exp01_slope03_rchgptq/configs/exp01.env)

## Why This Lane

PR #1948 reported two near-free wins:
- LeakyReLU-square slope `0.3` improved BPB versus the older `0.5` slope in their sweep.
- Reverse-Cholesky GPTQ is mathematically equivalent to the older `cholesky_inverse + chol(upper)` path, but faster in their benchmark.

On #2014, the expected effect is:
- slope `0.3`: possible BPB improvement
- reverse-Cholesky: mostly timing/compliance margin, not direct modeling improvement

## Static Check

Run this locally before cloud:

```bash
python3 frontier/lanes/pr2014_exp01_slope03_rchgptq/checks/check_exp01_lane.py
```

This does not train. It verifies the patch markers and runs a small reverse-Cholesky equivalence check if `torch` is available.

## Data

This lane uses the same CaseOps dataset as #2014. If you already have `/workspace/pr2014_data.env`, you can reuse it.

If not, generate an exp01 env file:

```bash
cd /workspace/parameter-golf
bash frontier/lanes/pr2014_exp01_slope03_rchgptq/bootstrap_workspace_data.sh
```

That writes `/workspace/pr2014_exp01_data.env`.

## Cheap Comparison

Run baseline #2014 first:

```bash
source /workspace/pr2014_data.env
NPROC_PER_NODE=1 \
SEEDS="42" \
MAX_WALLCLOCK_SECONDS=300 \
TTT_ENABLED=0 \
COMPRESSOR=brotli \
RUN_ROOT=/workspace/runs/pr2014_baseline_quant_smoke \
bash frontier/lanes/pr2014/run_8xh100.sh
```

Then run exp01:

```bash
source /workspace/pr2014_data.env
NPROC_PER_NODE=1 \
SEEDS="42" \
MAX_WALLCLOCK_SECONDS=300 \
TTT_ENABLED=0 \
COMPRESSOR=brotli \
RUN_ROOT=/workspace/runs/pr2014_exp01_quant_smoke \
bash frontier/lanes/pr2014_exp01_slope03_rchgptq/run_8xh100.sh
```

Use `/workspace/pr2014_exp01_data.env` instead of `/workspace/pr2014_data.env` if that is the env file you generated.

## Promotion Criteria

Promote only if:
- pre-quant or quantized BPB is neutral or better than the same-budget #2014 baseline
- artifact size remains under budget
- GPTQ timing is stable
- no score-before-update or CaseOps byte-accounting behavior changes

## Size Rescue

If a completed run is only slightly over the `16,000,000` byte cap, first install
the code minifier and re-run packaging from the saved full-precision checkpoint:

```bash
cd /workspace/parameter-golf
source /workspace/pr2014_data.env
source .venv/bin/activate
python -m pip install python-minifier

RUN_ROOT=/workspace/runs/pr2014_exp01_seed42 \
NPROC_PER_NODE=1 \
SEEDS="42" \
QUANTIZE_ONLY=1 \
TTT_ENABLED=0 \
COMPRESSOR=pergroup \
bash frontier/lanes/pr2014_exp01_slope03_rchgptq/run_8xh100.sh
```

Use the same `RUN_ROOT` that contains `seed_42/final_model.pt`; otherwise
`QUANTIZE_ONLY=1` cannot find the checkpoint.

If the package is still over budget, repackage with the size-safe overlay:

```bash
cd /workspace/parameter-golf
source /workspace/pr2014_data.env
source .venv/bin/activate

RUN_ROOT=/workspace/runs/pr2014_exp01_seed42 \
CONFIG_PATH=frontier/lanes/pr2014_exp01_slope03_rchgptq/configs/fit16mb.env \
NPROC_PER_NODE=1 \
SEEDS="42" \
QUANTIZE_ONLY=1 \
TTT_ENABLED=0 \
COMPRESSOR=pergroup \
bash frontier/lanes/pr2014_exp01_slope03_rchgptq/run_8xh100.sh
```

This disables only `AWQ_LITE_GROUP_TOP_K`; if BPB regresses materially, reject
the size-safe artifact rather than promoting it.

## Eval-Time Rescue

If the final TTT eval is slightly over `600s`, do not retrain first. Reuse the
saved quantized artifact and run eval-only with a shorter TTT window:

```bash
cd /workspace/parameter-golf
source /workspace/pr2014_data.env
source .venv/bin/activate

RUN_ROOT=/workspace/runs/pr2014_exp01_seed42 \
CONFIG_PATH=frontier/lanes/pr2014_exp01_slope03_rchgptq/configs/fasteval2816.env \
NPROC_PER_NODE=8 \
SEEDS="42" \
TTT_EVAL_ONLY=1 \
COMPRESSOR=pergroup \
bash frontier/lanes/pr2014_exp01_slope03_rchgptq/run_8xh100.sh
```

If `2816` still exceeds the budget, use the stronger `2560` fallback:

```bash
RUN_ROOT=/workspace/runs/pr2014_exp01_seed42 \
CONFIG_PATH=frontier/lanes/pr2014_exp01_slope03_rchgptq/configs/fasteval2560.env \
NPROC_PER_NODE=8 \
SEEDS="42" \
TTT_EVAL_ONLY=1 \
COMPRESSOR=pergroup \
bash frontier/lanes/pr2014_exp01_slope03_rchgptq/run_8xh100.sh
```

If the run is both over size and over eval time, use the combined safe overlay.
First repackage the saved full-precision checkpoint:

```bash
RUN_ROOT=/workspace/runs/pr2014_exp01_seed42 \
CONFIG_PATH=frontier/lanes/pr2014_exp01_slope03_rchgptq/configs/submission_safe.env \
NPROC_PER_NODE=8 \
SEEDS="42" \
QUANTIZE_ONLY=1 \
TTT_ENABLED=0 \
COMPRESSOR=pergroup \
bash frontier/lanes/pr2014_exp01_slope03_rchgptq/run_8xh100.sh
```

Then run TTT eval-only on the newly written quantized artifact:

```bash
RUN_ROOT=/workspace/runs/pr2014_exp01_seed42 \
CONFIG_PATH=frontier/lanes/pr2014_exp01_slope03_rchgptq/configs/submission_safe.env \
NPROC_PER_NODE=8 \
SEEDS="42" \
TTT_EVAL_ONLY=1 \
COMPRESSOR=pergroup \
bash frontier/lanes/pr2014_exp01_slope03_rchgptq/run_8xh100.sh
```

If the combined `2816` overlay still exceeds `600s`, repeat those two commands
with `configs/submission_safe2560.env`.

For official timing, run eval-only with `NPROC_PER_NODE=8`; a 1-GPU eval-only
run is useful for debugging but is not comparable to the leaderboard budget.

If exp01 is neutral/positive on 1-GPU, run one 8xH100 seed:

```bash
source /workspace/pr2014_data.env
NPROC_PER_NODE=8 \
SEEDS="42" \
RUN_ROOT=/workspace/runs/pr2014_exp01_seed42 \
bash frontier/lanes/pr2014_exp01_slope03_rchgptq/run_8xh100.sh
```

Only after a good one-seed 8xH100 result should we consider 3-seed confirmation.
