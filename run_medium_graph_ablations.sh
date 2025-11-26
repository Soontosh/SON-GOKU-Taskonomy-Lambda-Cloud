#!/usr/bin/env bash
#set -euo pipefail

echo "[`date`] Starting medium graph ablations"

# Medium subset root
export TASKONOMY_MEDIUM=/lambda/nfs/india-training/taskonomy_medium/reshaped

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

###############################################################################
# Regime 1: Fixed-τ schedule (as in the paper)
###############################################################################
echo "[`date`] Regime 1: fixed-τ schedule"

# 1) Threshold (baseline SON-GOKU)
echo "[`date`] [fixed] threshold"
SON_GOKU_INSTRUMENT=1 \
SON_GOKU_DUMP_DIR=experiments/graph_rules/medium/fixed/threshold/graphs \
python -m taskonomy_eval.runner \
  "${COMMON_ARGS[@]}" \
  --graph_rule threshold \
  --out_dir experiments/graph_rules/medium/fixed/threshold

# 2) kNN-symmetric
echo "[`date`] [fixed] knn (m=4)"
SON_GOKU_INSTRUMENT=1 \
SON_GOKU_DUMP_DIR=experiments/graph_rules/medium/fixed/knn_m4/graphs \
python -m taskonomy_eval.runner \
  "${COMMON_ARGS[@]}" \
  --graph_rule knn \
  --graph_knn_m 4 \
  --out_dir experiments/graph_rules/medium/fixed/knn_m4

# 3) Signed-only
echo "[`date`] [fixed] signed-only"
SON_GOKU_INSTRUMENT=1 \
SON_GOKU_DUMP_DIR=experiments/graph_rules/medium/fixed/signed/graphs \
python -m taskonomy_eval.runner \
  "${COMMON_ARGS[@]}" \
  --graph_rule signed \
  --out_dir experiments/graph_rules/medium/fixed/signed

# 4) Quantile / percentile (worst 20% of pairs)
echo "[`date`] [fixed] quantile p=0.2"
SON_GOKU_INSTRUMENT=1 \
SON_GOKU_DUMP_DIR=experiments/graph_rules/medium/fixed/quantile_p0.2/graphs \
python -m taskonomy_eval.runner \
  "${COMMON_ARGS[@]}" \
  --graph_rule quantile \
  --graph_quantile_p 0.2 \
  --out_dir experiments/graph_rules/medium/fixed/quantile_p0.2

###############################################################################
# Regime 2: Density-matched graphs
###############################################################################
echo "[`date`] Regime 2: density-matched graphs"

TARGET_DENSITY=0.15  # you can adjust this if desired

# 5) Threshold, density-matched
echo "[`date`] [density] threshold (target density=${TARGET_DENSITY})"
SON_GOKU_INSTRUMENT=1 \
SON_GOKU_DUMP_DIR=experiments/graph_rules/medium/density/threshold_d0.15/graphs \
python -m taskonomy_eval.runner \
  "${COMMON_ARGS[@]}" \
  --graph_rule threshold \
  --graph_density_target "${TARGET_DENSITY}" \
  --out_dir experiments/graph_rules/medium/density/threshold_d0.15

# 6) kNN, density-matched
echo "[`date`] [density] knn (target density=${TARGET_DENSITY})"
SON_GOKU_INSTRUMENT=1 \
SON_GOKU_DUMP_DIR=experiments/graph_rules/medium/density/knn_d0.15/graphs \
python -m taskonomy_eval.runner \
  "${COMMON_ARGS[@]}" \
  --graph_rule knn \
  --graph_density_target "${TARGET_DENSITY}" \
  --out_dir experiments/graph_rules/medium/density/knn_d0.15

# 7) Signed-only, natural density (reported as-is)
echo "[`date`] [density] signed-only (natural density)"
SON_GOKU_INSTRUMENT=1 \
SON_GOKU_DUMP_DIR=experiments/graph_rules/medium/density/signed_natural/graphs \
python -m taskonomy_eval.runner \
  "${COMMON_ARGS[@]}" \
  --graph_rule signed \
  --out_dir experiments/graph_rules/medium/density/signed_natural

# 8) Quantile, density-matched
echo "[`date`] [density] quantile (target density=${TARGET_DENSITY})"
SON_GOKU_INSTRUMENT=1 \
SON_GOKU_DUMP_DIR=experiments/graph_rules/medium/density/quantile_d0.15/graphs \
python -m taskonomy_eval.runner \
  "${COMMON_ARGS[@]}" \
  --graph_rule quantile \
  --graph_density_target "${TARGET_DENSITY}" \
  --out_dir experiments/graph_rules/medium/density/quantile_d0.15

echo "[`date`] All medium graph ablations finished."