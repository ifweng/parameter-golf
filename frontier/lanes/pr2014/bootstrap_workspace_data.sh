#!/usr/bin/env bash
set -euo pipefail

LANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LANE_DIR/../../.." && pwd)"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/data/datasets/fineweb10B_sp8192_caseops/datasets}"
RUN_ROOT="${RUN_ROOT:-$WORKSPACE_ROOT/runs/pr2014_seed42}"
PYTHON="${PYTHON:-python3}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
MAX_WORKERS="${MAX_WORKERS:-8}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-$REPO_ROOT/frontier/lanes/pr1787/scripts/prepare_caseops_dataset.py}"

echo "PR2014 workspace data bootstrap"
echo "repo_root=$REPO_ROOT"
echo "workspace_root=$WORKSPACE_ROOT"
echo "data_root=$DATA_ROOT"
echo "train_shards=$TRAIN_SHARDS"
echo "max_workers=$MAX_WORKERS"

if ! "$PYTHON" -c "import huggingface_hub" >/dev/null 2>&1; then
  "$PYTHON" -m pip install --upgrade huggingface-hub
fi

"$PYTHON" "$PREPARE_SCRIPT" \
  --out-root "$DATA_ROOT" \
  --train-shards "$TRAIN_SHARDS" \
  --max-workers "$MAX_WORKERS"

CASEOPS_DATA_PATH="$DATA_ROOT/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
TOKENIZER_PATH="$DATA_ROOT/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
ENV_PATH="$WORKSPACE_ROOT/pr2014_data.env"

cat >"$ENV_PATH" <<EOF
export CASEOPS_DATA_PATH="$CASEOPS_DATA_PATH"
export TOKENIZER_PATH="$TOKENIZER_PATH"
export RUN_ROOT="$RUN_ROOT"
EOF

echo
echo "Wrote $ENV_PATH"
echo
echo "PR2014 requires lrzip when COMPRESSOR=pergroup and PREQUANT_ONLY is not set:"
echo "sudo apt-get update && sudo apt-get install -y lrzip"
echo
echo "Next cheap 1-GPU pre-quant smoke:"
echo "source $ENV_PATH"
echo 'NPROC_PER_NODE=1 SEEDS="42" MAX_WALLCLOCK_SECONDS=180 PREQUANT_ONLY=1 COMPRESSOR=brotli RUN_ROOT=/workspace/runs/pr2014_1gpu_prequant_smoke bash frontier/lanes/pr2014/run_8xh100.sh'
echo
echo "Next 1-GPU quant smoke after pre-quant works:"
echo "source $ENV_PATH"
echo 'NPROC_PER_NODE=1 SEEDS="42" MAX_WALLCLOCK_SECONDS=300 TTT_ENABLED=0 COMPRESSOR=brotli RUN_ROOT=/workspace/runs/pr2014_1gpu_quant_smoke bash frontier/lanes/pr2014/run_8xh100.sh'
echo
echo "Final narrow 8xH100 confirmation:"
echo "source $ENV_PATH"
echo 'NPROC_PER_NODE=8 SEEDS="42 314 0" RUN_ROOT=/workspace/runs/pr2014_3seed bash frontier/lanes/pr2014/run_8xh100.sh'
