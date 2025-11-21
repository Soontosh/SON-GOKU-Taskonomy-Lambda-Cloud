# LAMBDA_SPIN_UP_GUIDE

Spin-up checklist for **Lambda Cloud + Taskonomy + SON-GOKU** on a **Virginia (us-east-1)** instance, using **WSL Ubuntu** as your local client.

This assumes:

- You’re on Windows with **WSL Ubuntu**.
- You’re using **Lambda Cloud On-Demand**.
- Your **filesystem and instance are in Virginia (`us-east-1`)**.

---

## 1. Create a filesystem (Virginia region)

1. Log into the **Lambda Cloud console**.
2. In the left sidebar, go to **Storage → Filesystems**.
3. Click **Create filesystem**.
4. Choose:
   - **Name:** e.g. `taskonomy`
   - **Region:** `us-east-1 – Virginia, USA`
5. Click **Create filesystem**.

Once attached to an instance, Lambda mounts the filesystem at:

```text
/lambda/nfs/<FILESYSTEM_NAME>
```

and also creates a **symlink in the `ubuntu` home directory** with the same name (e.g. `~/taskonomy`).

---

## 2. Launch an instance (Virginia + filesystem attached)

1. In the console, go to **Instances**.
2. Click **Launch instance**.
3. In the wizard:

   - **Instance type:** pick your GPU (e.g. `A10`, `A100`, etc.).
   - **Region:** `us-east-1 – Virginia, USA`.
   - **Image:** a recent **Lambda Stack (Ubuntu)** image.
   - **Filesystem:** select the filesystem you just created (`taskonomy`).
   - **SSH key:** choose the SSH key you’ve added to Lambda (next section).
4. Confirm terms and click **Launch instance**.

After boot, you’ll see the instance listed with:

- **IP address**
- **Attached filesystem** (in same region)

---

## 3. SSH from WSL Ubuntu into the instance

### 3.1 Generate (or reuse) an SSH key in WSL

In **WSL Ubuntu** terminal:

```bash
# Check for existing keys
ls ~/.ssh

# If you don't have an ed25519 key yet:
ssh-keygen -t ed25519 -C "lambda-cloud"
# (press Enter to accept default path, optionally add a passphrase)
```

Copy your **public** key:

```bash
cat ~/.ssh/id_ed25519.pub
```

### 3.2 Add SSH key to Lambda Cloud

1. In the console, go to **SSH keys** in the left sidebar.
2. Click **Add SSH key**.
3. Paste your `id_ed25519.pub` into the text box and give it a name (e.g. `wsl-ed25519`).
4. Click **Add SSH key**.

This key can now be attached to instances at launch.

### 3.3 Connect from WSL

Once the instance is **Running**:

1. Go to **Instances**, copy the IPv4 address.
2. From WSL:

   ```bash
   # If you used the default id_ed25519
   ssh ubuntu@<INSTANCE_IP>

   # Or explicitly specify the key:
   ssh -i ~/.ssh/id_ed25519 ubuntu@<INSTANCE_IP>
   ```

Lambda’s docs use exactly this pattern for connecting:

```bash
ssh -i '<SSH-KEY-FILE-PATH>' ubuntu@<INSTANCE-IP>
```

---

## 4. Install Linux packages on the instance

Once SSH’d in as `ubuntu`:

```bash
# Always start with updates
sudo apt-get update
sudo apt-get upgrade -y
```

Install tools we’ll need:

```bash
sudo apt-get install -y \
    git build-essential python3-venv python3-dev \
    aria2 tmux tree
```

- `aria2` is recommended by **Omnidata / omnitools** to support high-speed multi-connection downloads.
- You can verify the GPU is visible:

```bash
nvidia-smi
```

(Lambda tutorials often suggest this as a sanity check after instance launch.)

---

## 5. Create and activate a GPU-aware Python virtual environment

We want our venv to **see the CUDA-enabled PyTorch that comes with Lambda Stack**, not override it with a CPU-only wheel. The trick is to use `--system-site-packages`.

On the instance:

```bash
mkdir -p ~/venvs
python3 -m venv --system-site-packages ~/venvs/taskonomy-gpu

# Activate
source ~/venvs/taskonomy-gpu/bin/activate

# Update pip
python -m pip install --upgrade pip
```

You’ll know the venv is active when your prompt is prefixed with `(taskonomy-gpu)`.

---

## 6. Clone the repo (and configure GitHub SSH access)

### 6.1 Add your SSH key to GitHub

Generate a key:

```bash
ssh-keygen -t ed25519 -C "sapatapatiwork@gmail.com"
```

Press Enter to accept the default file location. Then, display the OpenSSH Public Key:

```bash
cat ~/.ssh/id_ed25519.pub
```

1. Copy the whole line (starts with `ssh-ed25519`)
2. On GitHub:

   - Go to **Settings → SSH and GPG keys → New SSH key**.
   - Paste the key, give it a name, save.

You can test from WSL:

```bash
ssh -T git@github.com
# Expected: "You've successfully authenticated, but GitHub does not provide shell access."
```

### 6.2 Clone onto the Lambda instance

<!--
SSH into the instance, activate your venv:

```bash
ssh ubuntu@<INSTANCE_IP>
source ~/venvs/taskonomy-gpu/bin/activate
cd ~
```
-->

Then clone the repo (SSH form is best for private access):

```bash
git clone git@github.com:Soontosh/SON-GOKU-Taskonomy-Lambda-Cloud.git
cd SON-GOKU-Taskonomy-Lambda-Cloud
```

(If you prefer HTTPS and the repo is public:
`git clone https://github.com/Soontosh/SON-GOKU-Taskonomy-Lambda-Cloud.git`)

### 6.3 Verify CUDA-enabled PyTorch via Lambda Stack

Because this venv was created with `--system-site-packages`, it can see the **GPU-enabled PyTorch installed by Lambda Stack**. You should **not** install `torch` via `pip` here (that risks overwriting it with a CPU-only wheel).

Check from inside `(taskonomy-gpu)`:

```bash
source ~/venvs/taskonomy-gpu/bin/activate

python - << 'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
PY
```

You want:

- `CUDA version:` to be non-`None`
- `CUDA available: True`
- `Device count: >= 1`

If CUDA is **not** available here:

- Double-check you launched a **GPU instance** with a **Lambda Stack** image.
- Make sure you’re not in a different venv.
- Avoid `pip install torch` in this env; that can overshadow Lambda’s CUDA build.

---

## 7. Install the repo as a package (`pip install -e .`)

From inside the repo:

```bash
cd ~/SON-GOKU-Taskonomy-Lambda-Cloud
source ~/venvs/taskonomy-gpu/bin/activate

pip install -e .
```

This uses the `pyproject.toml` to install `son_goku` in **editable mode**, so any local changes to the repo are picked up immediately.

Install a few extra runtime deps if needed:

```bash
pip install omnidata-tools numpy tqdm Pillow
```

---

## 8. Download + prepare Taskonomy data

We’ll use:

- **`omnitools.download`** to fetch Omnidata Taskonomy subsets.
- Your **`prepare_taskonomy_data.py`** helper script to:

  - Restructure into `data_root/split/building/domain/*.png`
  - Optionally create random train/val/test splits.

### 8.1 Install Omnidata tools

With your venv active:

```bash
pip install omnidata-tools
```

> ⚠️ If you use `--agree_all`, **you must** pass `--name` and `--email` or omnitools will raise a ValueError.

### 8.2 Place `prepare_taskonomy_data.py` in the repo

Save the `prepare_taskonomy_data.py` script into the repo root:

```bash
cd ~/SON-GOKU-Taskonomy-Lambda-Cloud
rm prepare_taskonomy_data.py
touch prepare_taskonomy_data.py
nano prepare_taskonomy_data.py
# paste the full script here, then save+exit
chmod +x prepare_taskonomy_data.py
```

No `sudo` is needed here: your filesystem is already mounted under `/lambda/nfs/<FILESYSTEM_NAME>` and writable by `ubuntu`.

### 8.3 Choose a filesystem path for raw + reshaped data

Your filesystem from step 1 is mounted at:

```text
/lambda/nfs/<FILESYSTEM_NAME>
```

and symlinked into your home as `~/<FILESYSTEM_NAME>`.

For example, if you named the filesystem `taskonomy`:

- Raw downloads: `/lambda/nfs/taskonomy`
- Reshaped data root: `/lambda/nfs/taskonomy/reshaped`

Optional home symlink:

```bash
ln -s /lambda/nfs/taskonomy/reshaped ~/taskonomy-reshaped
```

### 8.4 Run the data-prep script

Example (using the **debug** subset for quick tests):

```bash
cd ~/SON-GOKU-Taskonomy-Lambda-Cloud
source ~/venvs/taskonomy-gpu/bin/activate

python prepare_taskonomy_data.py \
  --download-root /lambda/nfs/taskonomy \
  --reshape-root  /lambda/nfs/taskonomy/reshaped \
  --subset medium \
  --download-split all \
  --domains all \
  --connections-total 32 \
  --name "Santosh Patapati" \
  --email "sapatapatiwork@gmail.com" \
  --agree_all \
  --train-frac 0.8 \
  --val-frac   0.1 \
  --test-frac  0.1 \
```

Notes:

- `--subset` can be `debug`, `tiny`, `medium`, `full`, `fullplus` per Omnidata’s starter dataset variants.
- `--domains` should always include `rgb` plus whatever tasks you train on (`depth_euclidean`, `normal`, `reshading`, etc.).
- `--train-frac`, `--val-frac`, `--test-frac` control how **buildings** are split into splits.

The script will:

- Call `omnitools.download` for your chosen subset & domains.

- Detect whether layout is `<dest>/<domain>/<component>/<building>` or `<dest>/<component>/<domain>/<building>`.

- Build a **Taskonomy-style tree**:

  ```text
  /lambda/nfs/taskonomy/reshaped/
    train/
      <building>/
        rgb/
        depth_euclidean/
        normal/
        reshading/
    val/
      ...
    test/
      ...
  ```

- Ensure that only views where **all requested tasks exist** are kept (multi-task samples).

---

## 9. Run training

You now have two complementary ways to train:

1. **The original SON-GOKU Taskonomy script** (kept exactly as-is).
2. **The new multi-method harness** (SON-GOKU, GradNorm, and future methods), which uses the same data/model for fair comparison.

### 9.1 Single-method training: original SON-GOKU script

This uses the existing SON-GOKU training/eval path and is ideal when you just want to reproduce the original results or do SON-GOKU-only experiments.

First, ensure the example script points at your reshaped data:

```bash
cd ~/SON-GOKU-Taskonomy-Lambda-Cloud/taskonomy_son_goku_eval
nano run_example.sh
```

Update the `--data_root` (and splits) to match your reshaped tree:

```bash
python -m taskonomy_eval.train_taskonomy \
  --data_root /home/ubuntu/taskonomy-reshaped \
  --split train --val_split val \
  --tasks depth_euclidean normal reshading \
  --resize 256 256 \
  --epochs 1 --batch_size 4 --lr 1e-3 \
  --refresh_period 16 --tau_kind log --tau_initial 1.0 --tau_target 0.25 \
  --ema_beta 0.9 --min_updates_per_cycle 1
```

> 🔎 This guide assumes your local repo already includes the small `TaskonomyDataset._path` patch you made earlier, which maps filenames like `domain_rgb` → `domain_depth_euclidean` / `domain_normal` / `domain_reshading`. Make sure those changes are committed or re-applied if you reclone.

Run training:

```bash
cd ~/SON-GOKU-Taskonomy-Lambda-Cloud/taskonomy_son_goku_eval
source ~/venvs/taskonomy-gpu/bin/activate
bash run_example.sh
```

### 9.2 Multi-method training: SON-GOKU vs GradNorm (and more)

For **fair comparisons** across methods (SON-GOKU, GradNorm, future algorithms), use the unified runner. It:

- Builds the **same Taskonomy dataset + splits** for all methods.
- Builds the **same multi-task model** (shared backbone + per-task heads).
- Uses the **same losses and metrics**.
- Only changes *how* each method schedules/weights tasks and updates gradients.

Example: run SON-GOKU and GradNorm back-to-back on the same split:

```bash
cd ~/SON-GOKU-Taskonomy-Lambda-Cloud
source ~/venvs/taskonomy-gpu/bin/activate

python -m taskonomy_eval.runner \
  --data_root /home/ubuntu/taskonomy-reshaped \
  --split train --val_split val \
  --tasks depth_euclidean normal reshading \
  --resize 256 256 \
  --epochs 10 --batch_size 8 --lr 1e-3 \
  --methods son_goku gradnorm \
  --seeds 0 1 \
  --refresh_period 32 --tau_kind log --tau_initial 1.0 --tau_target 0.25 \
  --ema_beta 0.9 --min_updates_per_cycle 1 \
  --gradnorm_alpha 1.5 --gradnorm_lr 0.025 \
  --out_dir experiments/taskonomy_compare
```

This will sequentially run:

- SON-GOKU with seeds 0 and 1.
- GradNorm with seeds 0 and 1.

Each `(method, seed)` combination gets its own directory under `experiments/taskonomy_compare/`, containing:

- `config.json` – the full experiment config.
- `val_metrics_epoch*.json` – per-epoch validation metrics.
- `<method>_seed<seed>.pt` – final model checkpoint.

You can add more methods later (e.g., PCGrad, AdaTask, NashMTL, etc.) by implementing them under `taskonomy_eval/methods/` and registering them in the runner.

### 9.3 Long-running jobs with tmux

Whether you’re using the original SON-GOKU script or the multi-method runner, it’s convenient to keep jobs alive after disconnecting.

```bash
tmux new -s taskonomy
cd ~/SON-GOKU-Taskonomy-Lambda-Cloud
source ~/venvs/taskonomy-gpu/bin/activate

# Example: SON-GOKU only
bash taskonomy_son_goku_eval/run_example.sh

# Example: SON-GOKU + GradNorm
python -m taskonomy_eval.runner \
  --data_root /home/ubuntu/taskonomy-reshaped \
  --split train --val_split val \
  --tasks depth_euclidean normal reshading \
  --resize 256 256 \
  --epochs 10 --batch_size 8 --lr 1e-3 \
  --methods son_goku gradnorm \
  --seeds 0 \
  --out_dir experiments/taskonomy_compare

# detach with: Ctrl+B then D
```

You can later reattach with:

```bash
tmux attach -t taskonomy
```

### 9.4 Training Many Methods With Logging and Error Handling

```bash
python -m taskonomy_eval.runner   --data_root ~/taskonomy-reshaped   --split train   --val_split test   --test_split test   --tasks depth_euclidean normal reshading   --methods mgda pcgrad cagrad adatask sel_update nashmtl fairgrad famo   --epochs 1   --batch_size 32   --out_dir experiments/all_method
s_smoke
```

---

## Summary

Once this is all scripted, spinning up a fresh environment on a new **Lambda Virginia instance** basically becomes:

1. Create filesystem (`taskonomy`) in `us-east-1`.
2. Launch instance in `us-east-1` and attach `taskonomy` + SSH key.
3. SSH from WSL → install packages → create `taskonomy-gpu` venv **with `--system-site-packages`** so it can see Lambda Stack’s CUDA-enabled PyTorch.
4. Clone repo → `pip install -e .` → `pip install omnidata-tools numpy tqdm`.
5. Run `prepare_taskonomy_data.py` into `/lambda/nfs/taskonomy/reshaped`.
6. Point training scripts at `/home/ubuntu/taskonomy-reshaped`:

   - Use `taskonomy_son_goku_eval/run_example.sh` for **SON-GOKU-only** runs.
   - Use `python -m taskonomy_eval.runner --methods son_goku gradnorm ...` for **multi-method comparisons**.
7. Use `tmux` if you want training to continue after disconnecting.
