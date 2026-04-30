#!/usr/bin/env bash
set -euo pipefail

LANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LANE_DIR/../../.." && pwd)"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/data/datasets/fineweb10B_sp8192_caseops/datasets}"
RUN_ROOT="${RUN_ROOT:-$WORKSPACE_ROOT/runs/pr1855_seed42}"
PYTHON="${PYTHON:-python3}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
MAX_WORKERS="${MAX_WORKERS:-8}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-$REPO_ROOT/frontier/lanes/pr1787/scripts/prepare_caseops_dataset.py}"

echo "PR1855 workspace data bootstrap"
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
ENV_PATH="$WORKSPACE_ROOT/pr1855_data.env"

cat >"$ENV_PATH" <<EOF
export CASEOPS_DATA_PATH="$CASEOPS_DATA_PATH"
export TOKENIZER_PATH="$TOKENIZER_PATH"
export RUN_ROOT="$RUN_ROOT"
EOF

echo
echo "Wrote $ENV_PATH"
echo
echo "PR1855 requires lrzip when COMPRESSOR=pergroup:"
echo "sudo apt-get update && sudo apt-get install -y lrzip"
echo
echo "Next single-seed 1-GPU benchmark command:"
echo "source $ENV_PATH"
echo 'NPROC_PER_NODE=1 SEEDS="42" bash frontier/lanes/pr1855/run_8xh100.sh'
echo
echo "After seed 42 looks good, run a narrow 8xH100 confirmation:"
echo "source $ENV_PATH"
echo 'NPROC_PER_NODE=8 RUN_ROOT=/workspace/runs/pr1855_3seed SEEDS="42 0 1234" bash frontier/lanes/pr1855/run_8xh100.sh'

