#!/usr/bin/env bash
set -Eeuo pipefail

# Every attack/model cell is a separate Python process. This gives a hard allocator/module
# boundary between cells; seed parallelism is handled inside the process via spawn.
ATTACKS=(pgd square problem_space_dummy)
MODELS=(mlp cnn lstm transformer)
GPU_PARTITIONS="${GPU_PARTITIONS:-0}"   # comma-separated GPU ids, or "cpu"
SEED_START="${SEED_START:-1}"
SEED_COUNT="${SEED_COUNT:-5}"
WORKERS="${WORKERS:-0}"

IFS=',' read -ra GPU_IDS <<< "${GPU_PARTITIONS}"
HYDRA_DEVICES="["
for i in "${!GPU_IDS[@]}"; do
  [[ "$i" -gt 0 ]] && HYDRA_DEVICES+=","
  if [[ "${GPU_IDS[$i]}" == "cpu" ]]; then
    HYDRA_DEVICES+="cpu"
  else
    HYDRA_DEVICES+="cuda:${GPU_IDS[$i]}"
  fi
done
HYDRA_DEVICES+="]"

if [[ "${WORKERS}" == "0" ]]; then WORKERS="${#GPU_IDS[@]}"; fi

SEEDS="["
for ((i=0; i<SEED_COUNT; i++)); do
  (( i > 0 )) && SEEDS+=","
  SEEDS+="$((SEED_START + i))"
done
SEEDS+="]"

for attack in "${ATTACKS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=== attack=${attack} model=${model} devices=${HYDRA_DEVICES} seeds=${SEEDS} ==="
    python -u main.py \
      action=evaluate_attack \
      attack="${attack}" \
      model="${model}" \
      gpu="${GPU_IDS[0]}" \
      seeds="${SEEDS}" \
      parallel_workers="${WORKERS}" \
      gpu_partitions="${HYDRA_DEVICES}" \
      hydra.job.name="${attack}_${model}" \
      || { echo "FAILED attack=${attack} model=${model}" >&2; exit 1; }
  done
done
