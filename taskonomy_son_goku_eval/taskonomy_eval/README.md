
# SON-GOKU × Taskonomy — Training & Evaluation Scaffold (PyTorch)

This folder provides a runnable scaffold to **train and evaluate** the SON-GOKU algorithm on the **Taskonomy** dataset.
It uses the standalone SON-GOKU scheduler you installed earlier and wires it to a simple UNet-style multi-task model.

## 0) Install
First, install your SON-GOKU package (from the previous zip) and then this project's dependencies:
```bash
# from the directory where you unzipped the earlier SON-GOKU codebase
pip install -e ./son_goku

# now install extras for Taskonomy evaluation
pip install torch torchvision pillow numpy
```

## 1) Get the Taskonomy data
The official repository points to Omnidata tools for downloading Taskonomy. One-liners (requires aria2):
```bash
# Install tools
sudo apt-get install -y aria2
pip install 'omnidata-tools'

# Example: download Taskonomy component (you can choose subset 'tiny' or 'medium' to start)
omnitools.download all --components taskonomy --subset tiny --split train val test \
  --dest /path/to/taskonomy_root --connections_total 16 --agree
```
The expected on-disk layout is:
```
taskonomy_root/
  train/<building>/{rgb,depth_euclidean,depth_zbuffer,normal,reshading,edge_occlusion,segment_semantic,...}/*.png
  val/<building>/{...}/*.png
  test/<building>/{...}/*.png
```

## 2) Run training + validation
```bash
python -m taskonomy_eval.train_taskonomy \
  --data_root /path/to/taskonomy_root \
  --split train --val_split val \
  --tasks depth_euclidean normal reshading \
  --resize 256 256 \
  --epochs 5 --batch_size 8 --lr 1e-3 \
  --refresh_period 32 --tau_kind log --tau_initial 1.0 --tau_target 0.25 \
  --ema_beta 0.9 --min_updates_per_cycle 1
```
Notes:
- Add `segment_semantic` to `--tasks` if you also downloaded semantic labels (set `--seg_classes` accordingly).
- Increase `--subset` at download time to `medium` or `fullplus` for stronger results.

## Metrics
- **Depth**: RMSE / MAE / AbsRel (on meters).
- **Normals**: mean / median angular error (deg), plus `<11.25°`, `<22.5°`, `<30°` accuracy.
- **Reshading**: MAE.
- **Edges**: BCE + F1 at 0.5 (if you include `edge_occlusion`/`edge_texture`).
- **Semantic segmentation**: mIoU (requires `--seg_classes`).

## SON-GOKU knobs (all exposed)
- `--refresh_period`, `--ema_beta`, `--min_updates_per_cycle`
- `--tau_kind`, `--tau_initial`, `--tau_target`, `--tau_warmup`, `--tau_anneal`

## Tips
- Start with the **tiny** or **medium** subset to validate your setup.
- If you want to use your own backbone, swap out `taskonomy_eval/models/mtl_unet.py`—the rest of the code is head-agnostic.
- To combine SON-GOKU with gradient surgery, pass a `gradient_transform` to the scheduler (see the base SON-GOKU README).
