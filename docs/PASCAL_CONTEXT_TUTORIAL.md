# Pascal-Context Tutorial

This guide walks through preparing the Pascal-Context dataset and running the
multi-task segmentation experiments that mirror the existing Taskonomy
pipelines. It assumes you already cloned this repository and installed it in
development mode via `pip install -e .`.

## 1. Prerequisites

* **Python 3.10+** with `pip`.
* **PyTorch** (CPU or CUDA). For a CPU-only setup you can install the wheel via:
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
  ```
* **NumPy** and **Pillow** are required for dataset preparation utilities:
  ```bash
  pip install numpy pillow
  ```
* **Disk space.** The full Pascal-Context release expands to ~13 GB. Symlinking
  (instead of copying) the assets keeps storage requirements low.

## 2. Download Pascal-Context assets

1. Grab the **PASCAL VOC 2010** images and split lists from the official site:
   * http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html
2. Download the **PASCAL-Context annotations** (e.g., `VOC2010_context.tar.gz`)
   from the authors’ project page: http://www.cs.stanford.edu/~roozbeh/pascal-context/
3. Extract both archives. After unpacking you should have a layout similar to:
   ```text
   /data/pascal_context/
     VOCdevkit/
       VOC2010/
         JPEGImages/
         ImageSets/Segmentation/
         ...
     context/
       train/
       val/
   ```
   The PNG masks inside `context/<split>/` share the same stem as the RGB image
   files.

If you already have VOC2010 elsewhere, you can simply place the `context/`
folder alongside it or create symlinks—the preparation script will locate both
components automatically.

## 3. Reshape data with `prepare_pascal_context_data.py`

The Taskonomy runners expect a directory layout where each sample sits inside a
split-specific folder and contains per-task subdirectories (`rgb/`,
`segment_semantic/`, …). Use the provided helper to generate that structure:

```bash
python prepare_pascal_context_data.py \
  --pascal-root /data/pascal_context \
  --output-root /data/pcontext_taskonomy \
  --copy-mode symlink \
  --splits train val
```

Key options:

* `--pascal-root` can point either to `VOCdevkit/VOC2010`, its parent directory,
  or any folder that contains both `VOCdevkit/VOC2010` and `context/`.
* `--copy-mode` controls whether RGB/mask files are copied or symlinked.
* `--limit` is handy for smoke tests—it caps the number of images per split.
* Add `--force` to remove any existing `--output-root` directory before
  reshaping.

After the script finishes you should see a layout such as
`/data/pcontext_taskonomy/train/2007_000033/rgb/2007_000033.jpg` and the
matching `segment_semantic/2007_000033.png`.

## 4. Quick dataset sanity check

Verify that the loader can discover the prepared data and that class statistics
look reasonable:

```python
from pascal_context_eval.datasets import PascalContextConfig, PascalContextDataset

dataset = PascalContextDataset(
    PascalContextConfig(
        root="/data/pcontext_taskonomy",
        split="train",
        resize=(256, 256),
    )
)
print(len(dataset), "samples")
print(dataset[0]["rgb"].shape, dataset[0]["segment_semantic"].shape)
print("num_classes:", dataset.num_classes, "ignore_index:", dataset.ignore_index)
print("class frequencies:", dataset.class_frequencies())
```

If you need to experiment with a custom class taxonomy, supply `class_map`
(a JSON file mapping raw IDs to contiguous labels) when creating
`PascalContextConfig`.

## 5. Train with `pascal_context_eval.runner`

The Pascal runner mirrors the Taskonomy entry point and plugs directly into the
multi-task method registry. A minimal segmentation-only run on CPU looks like:

```bash
python -m pascal_context_eval.runner \
  --data-root /data/pcontext_taskonomy \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda:0 \
  --resize 320 320 \
  --method son_goku \
  --out-dir runs/pcontext_son_goku
```

Notable arguments:

* `--tasks` selects which targets to optimize (defaults to
  `segment_semantic`).
* `--label-set` chooses the built-in taxonomy (`59` by default). Use `--class-map`
  to point at a JSON file when experimenting with custom label subsets.
* `--method` accepts any key from `pascal_context_eval.methods.METHOD_REGISTRY`,
  e.g., `gradnorm`, `mgda`, `pcgrad`, or `famo`.
* SON-GOKU-specific knobs are exposed via `--refresh-period`, `--tau-*`, and
  `--min-updates-per-cycle`.
* Outputs (config, metrics, checkpoint, and stdout log) are written under
  `--out-dir`.

## 6. Evaluate an existing checkpoint

To compute validation/test metrics without further training, reuse the runner
with zero epochs and point it to the stored weights:

```bash
python -m pascal_context_eval.runner \
  --data-root /data/pcontext_taskonomy \
  --epochs 0 \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda:0 \
  --method son_goku \
  --out-dir runs/pcontext_eval_only \
  --test-split test
```

After the run, drop the checkpoint into the output directory and call the
helper below to evaluate it:

```python
import torch
from pascal_context_eval.runner import build_datasets, ExperimentConfig, evaluate, build_model

cfg = ExperimentConfig(
    data_root="/data/pcontext_taskonomy",
    split="train",
    val_split="val",
    test_split="test",
    tasks=("segment_semantic",),
    resize=(320, 320),
    label_set="59",
    mask_root=None,
    class_map=None,
    ignore_index=255,
    epochs=0,
    batch_size=8,
    lr=1e-4,
    base_channels=64,
    num_workers=4,
    device="cuda:0",
    method="son_goku",
    seed=0,
    out_dir="runs/pcontext_eval_only",
)
train_dataset, val_dataset, test_dataset = build_datasets(cfg)
model, _ = build_model(cfg.tasks, seg_classes=train_dataset.num_classes)
model.load_state_dict(torch.load("runs/pcontext_eval_only/son_goku_seed0.pt"))
metrics = evaluate(model.cuda(),
                   torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False),
                   cfg.tasks,
                   train_dataset.num_classes,
                   train_dataset.ignore_index,
                   torch.device(cfg.device))
print(metrics)
```

This mirrors the internal validation logic and produces comparable mIoU
numbers.

## 7. Troubleshooting & tips

* **Missing files.** The dataset loader reports skipped image IDs with missing
  RGBs or masks. Check the `context/<split>/` directory to ensure PNG names
  match the JPEG stems.
* **Ignore label handling.** The runner forwards `--ignore-index` to both the
  CrossEntropy loss and mIoU metric, so unlabeled pixels (value `255` by default)
  are safely masked out.
* **Custom tasks.** To extend beyond segmentation, create additional targets
  inside your prepared directory (e.g., boundary masks) and register the new
  task head/loss in `pascal_context_eval.runner` similar to the Taskonomy setup.
* **Performance tweaks.** Increase `--num-workers`, enable mixed precision, or
  leverage CUDA devices as needed—everything reuses the Taskonomy
  implementations under the hood.

With these steps you can go from raw Pascal-Context archives to reproducible
multi-task training runs powered by SON-GOKU and the accompanying baselines.
