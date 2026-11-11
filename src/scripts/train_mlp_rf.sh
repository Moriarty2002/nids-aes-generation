#!/usr/bin/env bash
set -Eeuo pipefail
# Target-only training with stable artifacts; no purported SOTA defaults.
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SRC_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE="${BUDGET_PROFILE:-research}"
SEEDS="${SEEDS:-'[42,52,62,72,82]'}"
GPU="${GPU:-cpu}"
cd "$REPO_ROOT"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/runs/experiment_$(date -u +%Y%m%d_%H%M%S)_$($PYTHON_BIN -c 'import uuid; print(uuid.uuid4().hex[:8])')}"
common=("budget=$PROFILE" "seeds=$SEEDS" "gpu=$GPU" "gpu_partitions=[$GPU]" \
        "experiment_dir=$EXPERIMENT_DIR")
for model in mlp random_forest; do
  training_group=default
  [[ "$model" == random_forest ]] && training_group=sklearn
  "$PYTHON_BIN" -u "$SRC_ROOT/main.py" action=train_nids "model=$model" \
    "model/training=$training_group" "${common[@]}" "$@"
done
