#!/usr/bin/env bash

# Deterministic, synthetic memory benchmark for every registered method.
# Each method is executed in its own Python process to minimize cross-talk.

set -euo pipefail

METHODS=("$@")
if [ "${#METHODS[@]}" -eq 0 ]; then
  METHODS=(
    son_goku
    gradnorm
    pcgrad
    mgda
    cagrad
    famo
    adatask
  )
fi

TASKS=(
  depth_euclidean
  normal
  reshading
)

OUT_ROOT="${OUT_ROOT:-experiments/memory_isolation/$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-/tmp/taskonomy_synthetic}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MEASURE_STEPS="${MEASURE_STEPS:-64}"
WARMUP_STEPS="${WARMUP_STEPS:-32}"
SYNTH_DATASET_SIZE="${SYNTH_DATASET_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_ROOT}"

echo "[synthetic-memory] Writing per-method outputs to ${OUT_ROOT}"

for method in "${METHODS[@]}"; do
  RUN_DIR="${OUT_ROOT}/${method}"
  echo "[synthetic-memory] measuring ${method} -> ${RUN_DIR}"
  python3 -m taskonomy_eval.memory_cli \
    --data_root "${DATA_ROOT}" \
    --split train \
    --val_split val \
    --tasks "${TASKS[@]}" \
    --exp m1_peak \
    --methods "${method}" \
    --seeds 0 \
    --batch_size "${BATCH_SIZE}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --measure_steps "${MEASURE_STEPS}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --synthetic_data \
    --synthetic_dataset_size "${SYNTH_DATASET_SIZE}" \
    --synthetic_seed 0 \
    --refresh_period 32 \
    --ema_beta 0.9 \
    --tau_kind log \
    --tau_initial 1.0 \
    --tau_target 0.25 \
    --tau_warmup 0 \
    --tau_anneal 0 \
    --min_updates_per_cycle 1 \
    --out_dir "${RUN_DIR}"
done

echo "[synthetic-memory] Done. Individual summaries live under ${OUT_ROOT}"
