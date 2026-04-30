#!/usr/bin/env bash
set -euo pipefail

LANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LANE_DIR/../../.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$LANE_DIR/configs/pr1855.env}"
# shellcheck source=/dev/null
source "$CONFIG_PATH"

TORCHRUN="${TORCHRUN:-torchrun}"
PYTHON="${PYTHON:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SEEDS="${SEEDS:-42 0 1234}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
CASEOPS_DATA_PATH="${CASEOPS_DATA_PATH:-$DATA_DIR/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$LANE_DIR/upstream/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/frontier/workdirs/cloud/pr1855-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! -f "$LANE_DIR/upstream/train_gpt.py" ]]; then
  echo "missing upstream train_gpt.py at $LANE_DIR/upstream/train_gpt.py" >&2
  exit 2
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "missing tokenizer at $TOKENIZER_PATH" >&2
  exit 2
fi

if ! compgen -G "$CASEOPS_DATA_PATH/fineweb_train_*.bin" >/dev/null; then
  echo "missing CaseOps train shards under $CASEOPS_DATA_PATH" >&2
  echo "set CASEOPS_DATA_PATH to a prepared PR #1855 CaseOps dataset" >&2
  exit 2
fi

if ! compgen -G "$CASEOPS_DATA_PATH/fineweb_val_bytes_*.bin" >/dev/null; then
  echo "missing CaseOps validation byte sidecars under $CASEOPS_DATA_PATH" >&2
  echo "BPB accounting is not trusted without fineweb_val_bytes_*.bin" >&2
  exit 2
fi

if [[ "${COMPRESSOR:-}" == "pergroup" ]] && ! command -v lrzip >/dev/null 2>&1; then
  echo "missing lrzip; install it first or override COMPRESSOR=brotli for non-submission smoke tests" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
cp "$CONFIG_PATH" "$RUN_ROOT/pr1855.env"

echo "PR1855 frontier lane"
echo "repo_root=$REPO_ROOT"
echo "lane_dir=$LANE_DIR"
echo "run_root=$RUN_ROOT"
echo "caseops_data_path=$CASEOPS_DATA_PATH"
echo "tokenizer_path=$TOKENIZER_PATH"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "seeds=$SEEDS"
echo "compressor=$COMPRESSOR"

for seed in $SEEDS; do
  ARTIFACT_DIR="$RUN_ROOT/seed_${seed}"
  mkdir -p "$ARTIFACT_DIR"
  LOG_PATH="$RUN_ROOT/train_seed${seed}.log"

  echo "starting seed=$seed log=$LOG_PATH artifact_dir=$ARTIFACT_DIR"
  (
    # shellcheck source=/dev/null
    source "$CONFIG_PATH"

    RUN_ID="pr1855_seed${seed}" \
    DATA_DIR="$DATA_DIR" \
    DATA_PATH="$CASEOPS_DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    ARTIFACT_DIR="$ARTIFACT_DIR" \
    SEED="$seed" \
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" "$LANE_DIR/upstream/train_gpt.py"
  ) >"$LOG_PATH" 2>&1
done

"$PYTHON" "$LANE_DIR/scripts/collect_metrics.py" --run-root "$RUN_ROOT"

