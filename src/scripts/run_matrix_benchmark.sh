#!/usr/bin/env bash
set -Eeuo pipefail

# One dependency-ordered matrix over the existing modular workflow.
# Usage (from any directory):
#   BUDGET_PROFILE=smoke DATASET_PARQUET=/absolute/path/ciciomt.parquet \
#   TARGET_MODELS='mlp cnn random_forest' ATTACK_MATRIX='pgd nids_constrained_pointwise' \
#   bash src/scripts/run_matrix_benchmark.sh
# The smoke profile is a software integration check, not a research/SOTA claim.
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SRC_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE="${BUDGET_PROFILE:-smoke}"
GPU="${GPU:-cpu}"
SEEDS="${SEEDS:-'[1]'}"
# Optional paired-attack design: train independent frozen checkpoints once,
# then evaluate different attack RNG replicates against the same split/weights.
# Leave unset for the original per-seed model-training matrix.
SPLIT_SEED="${SPLIT_SEED:-}"
TARGET_MODEL_SEED="${TARGET_MODEL_SEED:-}"
SURROGATE_MODEL_SEED="${SURROGATE_MODEL_SEED:-}"
ATTACK_REPLICATE_SEEDS="${ATTACK_REPLICATE_SEEDS:-$SEEDS}"
WORKERS="${WORKERS:-1}"
PROGRESS="${PROGRESS:-auto}"
VERBOSE_COMMANDS="${VERBOSE_COMMANDS:-0}"
TARGET_MODELS="${TARGET_MODELS:-mlp cnn random_forest}"
ATTACK_MATRIX="${ATTACK_MATRIX:-pgd nids_constrained_pointwise}"
DETECTOR="${DETECTOR:-none}"
DATASET_PARQUET="${DATASET_PARQUET:-${REPO_ROOT}/data/ciciomt2024.parquet}"
# One versioned experiment, shared by every dependency in the matrix.
# Set EXPERIMENT_DIR for a new matrix. Already-published seed checkpoints
# cannot be retrained in place; reuse the selected directory with standalone
# evaluate_attack commands instead.
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/runs/experiment_$(date -u +%Y%m%d_%H%M%S)_$($PYTHON_BIN -c 'import uuid; print(uuid.uuid4().hex[:8])')}"
COMPARABLE_BUDGET_ID="${COMPARABLE_BUDGET_ID:-}"
# Optional post-research read-only descriptive reporting. Never runs on smoke.
AUTO_COMPARE="${AUTO_COMPARE:-0}"
AUTO_COMPARE_REFERENCE="${AUTO_COMPARE_REFERENCE:-}"

case "$PROGRESS" in auto|on|off) ;; *) echo 'PROGRESS must be auto, on or off' >&2; exit 2;; esac
case "$PROFILE" in smoke|research) ;; *) echo 'BUDGET_PROFILE must be smoke or research' >&2; exit 2;; esac
# One common split is essential when the surrogate and target have different
# fitting seeds. No implicit split substitution is allowed.
if [[ -n "$TARGET_MODEL_SEED" || -n "$SURROGATE_MODEL_SEED" ]]; then
  if [[ -z "$SPLIT_SEED" || -z "$TARGET_MODEL_SEED" || -z "$SURROGATE_MODEL_SEED" ]]; then
    echo 'Frozen paired runs require SPLIT_SEED, TARGET_MODEL_SEED, and SURROGATE_MODEL_SEED' >&2
    exit 2
  fi
fi
if [[ "$ATTACK_REPLICATE_SEEDS" != "$SEEDS" && -z "$TARGET_MODEL_SEED" ]]; then
  echo 'Separate ATTACK_REPLICATE_SEEDS requires frozen target and surrogate seed roles' >&2
  exit 2
fi
for value in "$SPLIT_SEED" "$TARGET_MODEL_SEED" "$SURROGATE_MODEL_SEED"; do
  if [[ -n "$value" && ! "$value" =~ ^[0-9]+$ ]]; then
    echo 'Frozen role seeds must be nonnegative integers' >&2
    exit 2
  fi
done
case "$DETECTOR" in none|mahalanobis) ;; *) echo 'DETECTOR must be none or mahalanobis' >&2; exit 2;; esac
if [[ ! -f "$DATASET_PARQUET" ]]; then
  echo "Missing active CICIoMT2024 source: $DATASET_PARQUET" >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c 'import hydra, torch, sklearn, pandas, pyarrow' >/dev/null 2>&1; then
  echo 'Missing runtime dependency: hydra-core/torch/scikit-learn/pandas/pyarrow are required' >&2
  exit 2
fi
# Do not pre-create the experiment manifest: first train_nids initializes it.
mkdir -p "$(dirname "$EXPERIMENT_DIR")"
source_extra=("dataset.params.path=$DATASET_PARQUET" "experiment_dir=$EXPERIMENT_DIR")
common=("gpu=$GPU" "gpu_partitions=[$GPU]" "parallel_workers=$WORKERS" "budget=$PROFILE" "progress=$PROGRESS" "${source_extra[@]}")
if [[ -n "$SPLIT_SEED" ]]; then common+=("seed_roles.split=$SPLIT_SEED"); fi
if [[ -n "$TARGET_MODEL_SEED" ]]; then common+=("seed_roles.target=$TARGET_MODEL_SEED"); fi
if [[ -n "$SURROGATE_MODEL_SEED" ]]; then common+=("seed_roles.surrogate=$SURROGATE_MODEL_SEED"); fi
surrogate_fit_seeds="$SEEDS"
target_fit_seeds="$SEEDS"
if [[ -n "$SURROGATE_MODEL_SEED" ]]; then surrogate_fit_seeds="[$SURROGATE_MODEL_SEED]"; fi
if [[ -n "$TARGET_MODEL_SEED" ]]; then target_fit_seeds="[$TARGET_MODEL_SEED]"; fi
if [[ -n "$COMPARABLE_BUDGET_ID" ]]; then
  # Only specify when independently verified comparable between experiments.
  common+=("evaluation.comparable_budget_id=$COMPARABLE_BUDGET_ID")
fi
# Validate/plan actual matrix invocations once; these are command counts,
# never an estimate of elapsed training/evaluation time.
read -r -a target_models <<< "$TARGET_MODELS"
read -r -a attack_names <<< "$ATTACK_MATRIX"
if [[ ${#target_models[@]} -eq 0 || ${#attack_names[@]} -eq 0 ]]; then
  echo 'TARGET_MODELS and ATTACK_MATRIX must each be nonempty' >&2; exit 2
fi
offline_attacks=()
for attack in "${attack_names[@]}"; do
  case "$attack" in
    nids_gan|transformer_gan|tdtce|a2pm) offline_attacks+=("$attack") ;;
    amd_gan) echo 'AMD-GAN original is a distributional stress/TSTR study, not per-flow feasible ASR; run the separate registered action=evaluate_distribution_stress for TSTR' >&2; exit 2 ;;
    pgd|nids_constrained_pointwise|bim|fgsm|jsma|cw|deepfool|trades) ;;
    *) echo "Unsupported attack for this matrix: $attack" >&2; exit 2 ;;
  esac
done
command_total=$((1 + ${#target_models[@]} + ${#offline_attacks[@]} +
                ${#target_models[@]} * ${#attack_names[@]}))
command_index=0
run() {
  local stage="$1" label="$2" rc
  shift 2
  command_index=$((command_index + 1))
  echo "Command $command_index/$command_total [$stage] $label seed_roles=split:${SPLIT_SEED:-worker},target:${TARGET_MODEL_SEED:-worker},surrogate:${SURROGATE_MODEL_SEED:-worker}"
  if [[ "$VERBOSE_COMMANDS" == 1 ]]; then printf '+ %q ' "$PYTHON_BIN" -u "$SRC_ROOT/main.py" "$@"; echo; fi
  if "$PYTHON_BIN" -u "$SRC_ROOT/main.py" "$@"; then
    return 0
  else
    rc=$?
    echo "FAILED command $command_index/$command_total [$stage] $label exit=$rc experiment_dir=$EXPERIMENT_DIR" >&2
    return "$rc"
  fi
}

cd "$REPO_ROOT"
echo "Matrix plan: profile=$PROFILE targets=${target_models[*]} attacks=${attack_names[*]} seeds=$SEEDS progress=$PROGRESS"
echo "Matrix commands: 1 surrogate + ${#target_models[@]} targets + ${#offline_attacks[@]} offline generators + $((${#target_models[@]} * ${#attack_names[@]})) evaluations = $command_total commands (not an ETA)"
echo 'Source preflight complete; CICIoMT2024 adapter enforces the fixed T=20 policy'

run surrogate 'independently trained authorized surrogate' action=train_nids model=jmila_surrogate model/training=default detector=none \
  "${common[@]}" "seeds=$surrogate_fit_seeds"

for model in "${target_models[@]}"; do
  training_group=default
  [[ "$model" == random_forest ]] && training_group=sklearn
  run target "model=$model" action=train_nids "model=$model" "model/training=$training_group" "detector=$DETECTOR" \
    "${common[@]}" "seeds=$target_fit_seeds"
done

# Offline generators use the authorized surrogate; none may train on target weights.
for attack in "${offline_attacks[@]}"; do
  run 'offline generator' "attack=$attack" action=train_attack model=jmila_surrogate model/training=default "attack=$attack" \
    detector=none "${common[@]}" "seeds=$ATTACK_REPLICATE_SEEDS"
done

for model in "${target_models[@]}"; do
  training_group=default
  [[ "$model" == random_forest ]] && training_group=sklearn
  for attack in "${attack_names[@]}"; do
    run evaluation "model=$model attack=$attack" action=evaluate_attack "model=$model" "model/training=$training_group" "attack=$attack" \
      "detector=$DETECTOR" "${common[@]}" "seeds=$ATTACK_REPLICATE_SEEDS"
  done
done
cat <<EOF
Finished matrix. Versioned experiment and all checkpoints/logs: $EXPERIMENT_DIR
Seed roles: split=${SPLIT_SEED:-per-worker} target=${TARGET_MODEL_SEED:-per-worker} surrogate=${SURROGATE_MODEL_SEED:-per-worker} attack_replicates=$ATTACK_REPLICATE_SEEDS
Optional legacy equal-budget comparator (not the certified native-configuration publication path):
  $PYTHON_BIN $SRC_ROOT/scripts/paired_compare.py RUN_A RUN_B --out paired.json
Do not claim comparable budgets or paired significance from identical seeds alone.
Legacy paired_compare.py has an unresolved zero-eligible strict mask admission edge case;
use registered native_recipe_report.py for research-native descriptive comparisons.
EOF

# Optional reporting is a convenience only: it does not train, query, or alter
# any completed science execution, and report failure cannot roll it back.
if [[ "$AUTO_COMPARE" == 1 ]]; then
  if [[ "$PROFILE" != research ]]; then
    echo 'comparison skipped: AUTO_COMPARE requires BUDGET_PROFILE=research (smoke remains diagnostic)'
  elif [[ "$(wc -w <<< "$ATTACK_MATRIX")" -lt 2 ]]; then
    echo 'comparison skipped: fewer than two compatible completed attack runs'
  else
    REPORT_DIR="$(dirname "$EXPERIMENT_DIR")/reports/$(basename "$EXPERIMENT_DIR")"
    # The matrix already uses a Python/Hydra-style numeric seed list. Parse it
    # without shell eval so explicit worker seed sets are passed unchanged.
    REPORT_SEEDS="$($PYTHON_BIN -c 'import ast,sys; values=ast.literal_eval(sys.argv[1]); assert isinstance(values,(list,tuple)); print(" ".join(str(int(n)) for n in values))' "$ATTACK_REPLICATE_SEEDS")"
    read -r -a REPORT_SEED_ARGS <<< "$REPORT_SEEDS"
    REPORT_REF_ARGS=()
    [[ -n "$AUTO_COMPARE_REFERENCE" ]] && REPORT_REF_ARGS=(--reference "$AUTO_COMPARE_REFERENCE")
    if PYTHONPATH="$SRC_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN"          "$SRC_ROOT/scripts/native_recipe_report.py"          --experiment-dir "$EXPERIMENT_DIR" --discover --mode attack          --seeds "${REPORT_SEED_ARGS[@]}" "${REPORT_REF_ARGS[@]}"          --out "$REPORT_DIR/native_comparison.json"; then
      echo "Research comparison report: $REPORT_DIR/native_comparison.json"
    else
      echo 'comparison report skipped/failed: completed evaluations remain valid and unchanged; rerun reporter manually with --reference when needed' >&2
    fi
  fi
fi
