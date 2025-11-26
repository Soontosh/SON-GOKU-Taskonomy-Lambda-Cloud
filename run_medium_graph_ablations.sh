#!/usr/bin/env bash
# NOTE: we intentionally do NOT use `set -e` so that all 8 jobs run even if some fail.
set -uo pipefail

echo "[`date`] Starting medium graph ablations"

# Medium subset root
export TASKONOMY_MEDIUM=/lambda/nfs/india-training/taskonomy_medium/reshaped

# Base experiment dir + status log
BASE_EXP_DIR=experiments/graph_rules/medium
STATUS_LOG="$BASE_EXP_DIR/graph_ablation_status.log"
mkdir -p "$BASE_EXP_DIR"

echo "============================================" >> "$STATUS_LOG"
echo "[`date`] New graph ablation run started" >> "$STATUS_LOG"

# Activate venv and go to repo root
cd ~/SON-GOKU-Taskonomy-Lambda-Cloud
source ~/venvs/taskonomy-gpu/bin/activate

# Common training args (5 epochs, batch 32, all tasks)
COMMON_ARGS=(
  --data_root "$TASKONOMY_MEDIUM"
  --split train
  --val_split val
  --test_split test
  --tasks all
  --resize 256 256
  --epochs 5
  --batch_size 32
  --lr 1e-3
  --methods son_goku
  --seeds 0 1 2
  --refresh_period 32
  --tau_kind log
  --tau_initial 1.0
  --tau_target 0.25
  --ema_beta 0.9
  --min_updates_per_cycle 1
  --log_train_every 64
)

run_job () {
  local label="$1"; shift
  local dump_dir="$1"; shift
  local out_dir="$1"; shift

  # Ensure dirs exist
  mkdir -p "$out_dir"
  mkdir -p "$dump_dir"

  echo "[`date`] START $label" | tee -a "$STATUS_LOG"

  SON_GOKU_INSTRUMENT=1 \
  SON_GOKU_DUMP_DIR="$dump_dir" \
  python -m taskonomy_eval.runner \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --out_dir "$out_dir"

  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[`date`] DONE  $label : SUCCESS" | tee -a "$STATUS_LOG"
  else
    echo "[`date`] DONE  $label : FAIL (exit code $rc)" | tee -a "$STATUS_LOG"
  fi

  # Always continue to next job
  return 0
}

###############################################################################
# Regime 1: Fixed-τ schedule (as in the paper)
###############################################################################
echo "[`date`] Regime 1: fixed-τ schedule"

# 1) Threshold (baseline SON-GOKU)
run_job \
  "fixed_threshold" \
  "$BASE_EXP_DIR/fixed/threshold/graphs" \
  "$BASE_EXP_DIR/fixed/threshold" \
  --graph_rule threshold

# 2) kNN-symmetric (m=4)
run_job \
  "fixed_knn_m4" \
  "$BASE_EXP_DIR/fixed/knn_m4/graphs" \
  "$BASE_EXP_DIR/fixed/knn_m4" \
  --graph_rule knn \
  --graph_knn_m 4

# 3) Signed-only
run_job \
  "fixed_signed" \
  "$BASE_EXP_DIR/fixed/signed/graphs" \
  "$BASE_EXP_DIR/fixed/signed" \
  --graph_rule signed

# 4) Quantile / percentile (worst 20% of pairs)
run_job \
  "fixed_quantile_p0.2" \
  "$BASE_EXP_DIR/fixed/quantile_p0.2/graphs" \
  "$BASE_EXP_DIR/fixed/quantile_p0.2" \
  --graph_rule quantile \
  --graph_quantile_p 0.2

###############################################################################
# Regime 2: Density-matched graphs
###############################################################################
echo "[`date`] Regime 2: density-matched graphs"

TARGET_DENSITY=0.15  # you can adjust this if desired

# 5) Threshold, density-matched
run_job \
  "density_threshold_d0.15" \
  "$BASE_EXP_DIR/density/threshold_d0.15/graphs" \
  "$BASE_EXP_DIR/density/threshold_d0.15" \
  --graph_rule threshold \
  --graph_density_target "${TARGET_DENSITY}"

# 6) kNN, density-matched
run_job \
  "density_knn_d0.15" \
  "$BASE_EXP_DIR/density/knn_d0.15/graphs" \
  "$BASE_EXP_DIR/density/knn_d0.15" \
  --graph_rule knn \
  --graph_density_target "${TARGET_DENSITY}"

# 7) Signed-only, natural density (reported as-is)
run_job \
  "density_signed_natural" \
  "$BASE_EXP_DIR/density/signed_natural/graphs" \
  "$BASE_EXP_DIR/density/signed_natural" \
  --graph_rule signed

# 8) Quantile, density-matched
run_job \
  "density_quantile_d0.15" \
  "$BASE_EXP_DIR/density/quantile_d0.15/graphs" \
  "$BASE_EXP_DIR/density/quantile_d0.15" \
  --graph_rule quantile \
  --graph_density_target "${TARGET_DENSITY}"

echo "[`date`] All medium graph ablations finished." | tee -a "$STATUS_LOG"