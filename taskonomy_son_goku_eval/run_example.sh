#!/usr/bin/env bash
set -e
python -m taskonomy_eval.train_taskonomy \
  --data_root /data/taskonomy \
  --split train --val_split val \
  --tasks depth_euclidean normal reshading \
  --resize 256 256 \
  --epochs 1 --batch_size 4 --lr 1e-3 \
  --refresh_period 16 --tau_kind log --tau_initial 1.0 --tau_target 0.25 \
  --ema_beta 0.9 --min_updates_per_cycle 1
