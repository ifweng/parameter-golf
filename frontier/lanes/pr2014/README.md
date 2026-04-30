# PR2014 Frontier Lane

This lane is the current clean open frontier target.

Target:
- upstream PR: [openai/parameter-golf#2014](https://github.com/openai/parameter-golf/pull/2014)
- upstream head commit: `c9843c97dc6d24b7a806ef3a51effa7b26d67a97`
- track: `Track B`
- expected 3-seed mean: `1.05759252` post-TTT BPB
- expected 3-seed loss: `2.31441043` nats
- expected artifact: about `15,982,485` bytes mean, max `15,984,387`
- expected hardware: `8xH100 80GB SXM`

## Layout

- `upstream/`: frozen copy of the PR #2014 record folder, including readable `train_gpt.py`, CaseOps prep code, tokenizer, upstream logs, `requirements.txt`, and `submission.json`
- `configs/pr2014.env`: environment surface for reproducing the #2014 stack
- `run_8xh100.sh`: cloud runner for one or more seeds; accepts `NPROC_PER_NODE=1` for budget smoke runs
- `bootstrap_workspace_data.sh`: downloads the shared SP8192 CaseOps dataset to `/workspace`
- `checks/check_pr2014_lane.py`: local static and lightweight compliance sanity checks
- `scripts/collect_metrics.py`: wrapper around the shared metrics parser

## What We Are Reproducing

PR #2014 stacks on top of the accepted PR #1855 family and adds:
- progressive context growth: `1024@0.100,2048@0.700,3072@1.000`
- final train/eval/TTT context at `3072`
- full validation-tail accounting via `EVAL_INCLUDE_TAIL=1`
- long-context TTT with `TTT_MASK=no_qv`, `TTT_Q_LORA=0`, `TTT_V_LORA=0`
- softer local TTT updates with `TTT_LOCAL_LR_MULT=0.75`
- short-document score-first TTT chunks: `256:8,2000:24`
- one phased TTT pass with `PHASED_TTT_PREFIX_DOCS=2500`
- `QK_GAIN_INIT=5.25`
- AWQ-lite plus asymmetric logit rescale on the #1855 quant/compression stack

The important compliance point is that #2014 keeps `val_tokens == target_tokens == 47,853,343` in all three logs. The short-document TTT schedule changes chunk size only; each chunk is scored before the update is applied.

## Local Static Check

Run this before spending cloud time:

```bash
python3 frontier/lanes/pr2014/checks/check_pr2014_lane.py
```

This does not train. It verifies provenance, published metrics, CaseOps roundtrip behavior, tokenizer vocab size if `sentencepiece` is available, BOS-safe SmearGate, #2014 config knobs, and full validation target coverage in the upstream logs.

## Cloud Data Bootstrap

On the cloud box:

```bash
bash frontier/lanes/pr2014/bootstrap_workspace_data.sh
```

This writes `/workspace/pr2014_data.env` with the exact `CASEOPS_DATA_PATH`, `TOKENIZER_PATH`, and default `RUN_ROOT`.

PR #2014 requires the `lrzip` system binary when `COMPRESSOR=pergroup` and `PREQUANT_ONLY` is not set:

```bash
sudo apt-get update
sudo apt-get install -y lrzip
```

## Cheap 1-GPU Smoke Commands

Use pre-quant first to validate data, kernels, schedule activation, logging, and metric collection without needing `lrzip`:

```bash
source /workspace/pr2014_data.env
NPROC_PER_NODE=1 \
SEEDS="42" \
MAX_WALLCLOCK_SECONDS=180 \
PREQUANT_ONLY=1 \
COMPRESSOR=brotli \
RUN_ROOT=/workspace/runs/pr2014_1gpu_prequant_smoke \
bash frontier/lanes/pr2014/run_8xh100.sh
```

Then run a cheap quant smoke without TTT:

```bash
source /workspace/pr2014_data.env
NPROC_PER_NODE=1 \
SEEDS="42" \
MAX_WALLCLOCK_SECONDS=300 \
TTT_ENABLED=0 \
COMPRESSOR=brotli \
RUN_ROOT=/workspace/runs/pr2014_1gpu_quant_smoke \
bash frontier/lanes/pr2014/run_8xh100.sh
```

## 8xH100 Confirmation Command

Use this only after 1-GPU smoke runs and compliance checks pass:

```bash
source /workspace/pr2014_data.env
NPROC_PER_NODE=8 \
SEEDS="42 314 0" \
RUN_ROOT=/workspace/runs/pr2014_3seed \
bash frontier/lanes/pr2014/run_8xh100.sh
```

The runner writes one artifact directory per seed and then runs:

```bash
python3 frontier/lanes/pr2014/scripts/collect_metrics.py --run-root "$RUN_ROOT"
```

The metrics parser writes both:
- `metrics_summary.json`
- `metrics_summary.md`

The Markdown table includes loss, BPB, artifact bytes, code bytes, wallclock, TTT eval time, and validation target coverage.

## Acceptance Criteria

Treat reproduction as successful only if:
- train wallclock is under `600s`
- TTT eval wallclock is under `600s`
- artifact is under `16,000,000` bytes
- post-TTT BPB is directionally near upstream `1.05759`
- `val_tokens == target_tokens == 47,853,343`
- CaseOps byte sidecar is used for BPB accounting
- SmearGate masks BOS positions in both train and TTT paths
- no pre-score validation-token optimization is introduced

## Improvement Queue

Do not mix these into the reproduction baseline. Test them as separate lanes or config overlays after #2014 reproduces:
- GPTQ reverse-Cholesky from PR #1948. #2014 still appears to use the older `torch.cholesky_inverse(torch.linalg.cholesky(H))` path, so this may buy timing margin for longer GPTQ reserve or safer compression.
- LeakyReLU-square slope `0.3` from PR #1948. #2014 still appears to use `negative_slope=0.5`; the slope change is low-byte and easy to A/B.
- Slightly stricter GPTQ reserve, for example `8s` or `12s`, only if the 8xH100 step count still leaves enough BPB margin. This improves compliance comfort but may cost trained steps.
- Short-doc TTT schedule variants, for example `128:8,512:16,2000:24`, if 1-GPU eval timing suggests unused headroom.
- Progressive context schedule variants, for example moving the 2k to 3k transition earlier. This is likely the most promising modeling-side search, but it needs real H100 validation.

Avoid n-gram tilt or PPM-like mixtures unless the precompute and scoring path is timer-inclusive and clearly legal under issue #1017.
