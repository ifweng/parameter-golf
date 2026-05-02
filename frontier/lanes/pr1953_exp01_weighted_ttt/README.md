# PR #1953 Exp01 Weighted TTT

This lane adds one eval-time adaptation change on top of the frozen PR #1953 baseline:

- baseline lane: `frontier/lanes/pr1953`
- upstream PR: <https://github.com/openai/parameter-golf/pull/1953>
- baseline reported BPB: `1.05855370`
- experiment: continuous loss-weighted local LoRA TTT updates

The frozen PR #1953 lane is not edited. This lane owns the experiment.

## Method

PR #1953 uses score-first TTT with uniform local update weighting:

```text
score chunk -> accumulate BPB -> update LoRA with each active doc weighted equally
```

This experiment keeps the same score-before-update order but changes the local update:

```text
score chunk -> accumulate BPB -> compute detached per-doc chunk losses -> update LoRA with smooth loss-proportional weights
```

The default weight rule is:

```text
raw_weight = (per_doc_loss / running_loss_ref) ** alpha
weight = normalize_active_mean_to_1(raw_weight)
weight = clamp(weight, min_w, max_w)
```

Default knobs:

```bash
TTT_LOSS_WEIGHTED_ENABLED=1
TTT_LOSS_WEIGHT_ALPHA=0.5
TTT_LOSS_WEIGHT_MIN=0.5
TTT_LOSS_WEIGHT_MAX=1.75
TTT_LOSS_WEIGHT_EMA=0.98
```

This is different from the failed binary loss-gated TTT attempt. No chunks are dropped; easy chunks still update, just less, and hard chunks are capped.

## Compliance Posture

- Uses only losses from a chunk after that chunk has already been scored.
- Weighted gradients affect only future chunks.
- Does not change the base model architecture or artifact contents.
- Does not add n-gram tilt, target-token-gated features, or validation-derived precomputation.
- Logs `ttt_loss_weighted:` at startup and `lw:` in TTT progress lines when active.

## Local Check

```bash
python3 frontier/lanes/pr1953_exp01_weighted_ttt/checks/check_exp01_lane.py
```

## Cheap Smoke

```bash
source /workspace/pr1953_data.env
NPROC_PER_NODE=1 \
SEEDS="42" \
MAX_WALLCLOCK_SECONDS=120 \
PREQUANT_ONLY=1 \
COMPRESSOR=brotli \
RUN_ROOT=/workspace/runs/pr1953_exp01_weighted_prequant_smoke \
bash frontier/lanes/pr1953_exp01_weighted_ttt/run_8xh100.sh
```

## Cheapest Useful Test

If a PR #1953 seed-42 artifact already exists, run eval-only:

```bash
source /workspace/pr1953_data.env
TTT_EVAL_ONLY=1 \
ARTIFACT_ROOT=/workspace/runs/pr1953_seed42 \
RUN_ROOT=/workspace/runs/pr1953_exp01_weighted_evalonly_seed42 \
NPROC_PER_NODE=8 \
SEEDS="42" \
bash frontier/lanes/pr1953_exp01_weighted_ttt/run_8xh100.sh
```

## Full Seed 42

```bash
source /workspace/pr1953_data.env
NPROC_PER_NODE=8 \
SEEDS="42" \
RUN_ROOT=/workspace/runs/pr1953_exp01_weighted_seed42 \
bash frontier/lanes/pr1953_exp01_weighted_ttt/run_8xh100.sh
```

Promotion gate:

- beat reproduced PR #1953 seed-42 post-TTT BPB
- keep eval under `600s`
- keep artifact under `16,000,000` bytes
- show `lw:` progress logs, proving the weighted update path ran
