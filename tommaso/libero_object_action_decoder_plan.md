# LIBERO-Object Action Decoder Training Plan (ETHRC unified backbone, AWS A100)

Forward-looking plan to train a single LIBERO-object action decoder on top of the
ETHRC-finetuned Cosmos backbone (single unified model trained on all 4 LIBERO
suites combined) and verify it in the LIBERO simulator. Companion to the prior
bi_yams plan; reuses the same training stack on `/home/ethrc/Desktop/mimic-video`.

**Reviewed via `/plan-eng-review` 2026-05-09.** All locked-in review decisions
are reflected in the steps below; see `## GSTACK REVIEW REPORT` at the bottom.

---

## TL;DR

- **Step 0 (today, local 5090):** Run upstream `object_full` sim eval as a
  pipeline smoke test using the already-on-disk backbone + action decoder.
  Gives us (a) confirmation eval.sh works end-to-end and (b) a calibration
  anchor for the AWS run.
- **Backbone:** ETHRC unified Cosmos finetune,
  `s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-libero/outputs/posttraining/video2world_lora/2b_libero_cosmos/checkpoints/model/iter_000007000.pt`.
  ~12 GB raw → fuse to ~4 GB. **Hard gate:** verify `lora_alpha=32` from S3
  `config.yaml` before fusing (script hardcodes 32; mismatch = silently wrong
  weights).
- **Data:** S3 pre-regenerated HDF5s at
  `s3://…/datasets/libero_cosmos/all_episodes/*suite=libero_object*.hdf5`
  (~500 eps, ~3 GB). Matched visuals to the backbone's training distribution.
  Run `process_libero.py` → zarrs, then `precompute_t5.py`. Friend handles.
- **Hardware:** AWS, 8× A100 80 GB. `bsz=128 global` → micro=16/GPU.
- **Training:** 50K iters (not 80K — matches upstream eval.sh checkpoint
  iters of ~50K).
- **Eval:** vanilla LIBERO sim eval (`eval/libero/eval.sh`) restricted to
  `libero_object` suite, **with both** the new ETHRC-backbone decoder
  AND the upstream-backbone decoder (already on disk) for direct A/B
  calibration.
- **Scope discipline:** one suite (object), one new training run, one A/B in
  sim, no real-robot data, no extra ablations.

---

## Decisions locked in

| Knob | Choice | Source |
|---|---|---|
| Suite | `libero_object` only | user |
| Data fraction | `_full` (~500 eps) | user |
| Backbone | ETHRC unified `iter_000007000.pt` (will fuse) | user |
| Hardware | AWS 8× A100 80 GB | user |
| Eval | LIBERO sim eval, object suite only, **+ upstream baseline (already on disk)** | review 5A |
| Training data path | **S3 regen HDF5s (matched visuals)**, not the official mimic-video pipeline | review T-1A |
| Step 0 local smoke test | **Yes — full upstream eval on 5090 before AWS** | review T-2A |
| `max_iter` | **50,000** (matches upstream eval.sh checkpoint iters) | review 2 |
| `save_iter` | 5000 — but verify the override beats `world2action.py:105` hardcode AND edit the hardcode | review 1A |
| LoRA alpha during fuse | **Hard-gated on S3 `config.yaml`** | review 4 |
| Validation during training | disabled (libero block in `world2action.py`) | repo |
| Optimizer | `fusedadamw`, lr ≈ 1e-4 (sweep midpoint) | repo default |
| Cross-attention layer | `xattn_layer_idx=20` | repo default |
| Global batch size | 128 | repo default |

Open knobs deferred to launch time: exact A100 instance shape (p4d.24xlarge
is the obvious target), checkpoint cadence over 50K (every 5K → 10 saves).

---

## Inputs

### Already on the local 5090 box (no action)

```
model/checkpoints/text_encoder/t5-11b/                                                      # ~45 GB
model/checkpoints/video_backbone/tokenizer/tokenizer.pth                                    # Wan VAE
model/checkpoints/video_backbone/v2w_pretrained_cosmos.pt                                   # base
model/checkpoints/video_backbone/v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused.pt
model/checkpoints/action_decoder/w2a_libero_object_full_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128_iter_000050274.pt
eval/libero/LIBERO/                                                                          # LIBERO benchmark installed (friend, 2026-05-09)
eval/libero/LIBERO/libero/datasets/                                                          # ~15 GB, libero_object + libero_goal HF demos
```

The two action-decoder + backbone lines — upstream backbone + upstream object_full
decoder — let us run eval.sh out of the box for Step 0 and as the baseline
arm of the final A/B. No new training needed for the baseline.

The LIBERO install + HF datasets came from the friend running README steps 1+2
locally. Step 0 sim-eval is unblocked. The HF datasets at `eval/libero/LIBERO/libero/datasets/`
are NOT used in Path B for training (we use S3 HDF5s instead), but LIBERO benchmark
does load task suites from them at sim-eval runtime, so they stay on disk.

### To download (split: local 5090 for data prep, AWS for training)

Updated 2026-05-09: friend already did README steps 1+2 (LIBERO install +
HF demos) on the 5090. Path B locked in — we ignore the HF demos for
training and use S3 HDF5s instead. Data prep (Step 2) runs on the 5090
since LIBERO + T5-11B are already there. AWS gets the finished zarrs.

**On the 5090:**

1. **LIBERO-object dataset (pre-regenerated)** — primary path per review T-1A:
   ```
   s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-libero/datasets/libero_cosmos/all_episodes/*suite=libero_object*.hdf5
   ```
   500 episodes, ~3 GB. **Matched-visuals** to the backbone's training
   distribution (NVIDIA postprocessed). Skip the official LIBERO regenerate
   step — go directly to `process_libero.py`.

**On the AWS instance (smaller list now):**

1. **ETHRC unified backbone** (LoRA, post-training output):
   ```
   s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-libero/outputs/posttraining/video2world_lora/2b_libero_cosmos/checkpoints/model/iter_000007000.pt
   ```
   Size: 11.9 GB. Will be fused → ~4 GB → `v2w_libero_cosmos_unified_iter_000007000_fused.pt`.

2. **Run config (for alpha verification)** — small, but **mandatory before
   fusing**:
   ```
   s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-libero/outputs/posttraining/video2world_lora/2b_libero_cosmos/config.yaml
   ```

3. **LIBERO benchmark install** — needed for sim eval (Step 6). On a fresh
   AWS instance, replicate the friend's local install:
   ```bash
   cd /workspace/mimic-video/eval/libero
   uv pip install -r LIBERO/requirements.txt
   uv pip install -e LIBERO
   ```
   And rsync `eval/libero/LIBERO/libero/datasets/` (15 GB) from the 5090,
   OR re-run `download_libero_datasets.py --use-huggingface` on AWS (faster
   on a fat AWS pipe).

4. **(NOT downloading)** `datasets/libero_cosmos_mp4/`, T5-11B, raw S3 HDF5s.
   T5 embeddings are already baked into the zarrs from Step 2.3 on the 5090.

### What does **not** need to move

- bi_yams / carton-box / towel real-robot datasets — out of scope.
- The other 3 LIBERO suites (spatial / goal / 10) — out of scope.

---

## Step 0 — Local 5090 upstream-eval smoke test (do this TODAY, before AWS)

Goal: prove `eval.sh` runs end-to-end against an already-trained model and
get a calibration number for the upstream paper authors' decoder.

```bash
cd /home/ethrc/Desktop/mimic-video
. model/scripts/env_setup.sh                              # 5090 sm_120 workarounds OK here

# LIBERO install already done by friend (README steps 1+2, 2026-05-09):
#   eval/libero/LIBERO/                       # benchmark installed
#   eval/libero/LIBERO/libero/datasets/       # 15 GB HF demos
# If `python -c "import libero"` fails, re-run from eval/libero/:
#   uv pip install -r LIBERO/requirements.txt
#   uv pip install -e LIBERO

cd eval/libero

# Edit eval.sh:
#   line 1   — GPUS=(0)                  # only the 5090
#   line 8   — for i in $(seq 0 1)       # one parallel slot only (5090 has 32 GB)
#   line 29  — checkpoint_dir=/home/ethrc/Desktop/mimic-video/model/checkpoints
#   line 124 — models=(object_full)      # libero_object only, upstream backbone
#   line 128 — stop_steps=(0 5 10)       # cap to 3 values for smoke test (saves ~12x)
#
# Optional further cap: edit eval/libero/run.py to lower trials/task from 50 → 5.

bash eval.sh
```

**Acceptance**:
- eval.sh starts without import errors (LIBERO benchmark loads, run.py loads
  the action decoder + backbone).
- At least one task produces a sim-eval `success_rate` in `results/`.
- We get a number for upstream `object_full` on `libero_object`. This number
  is our calibration anchor — **the AWS-trained ETHRC decoder needs to be
  in the same ballpark or better.**

If eval.sh fails: STOP the AWS plan. The eval pipeline has a bug we need to
fix before training a new decoder is meaningful. **Outside voice was right
to flag this — debugging eval after a 16h AWS run would be much more
expensive.**

If eval.sh succeeds: proceed to Step 1.

---

## Step 1 — Backbone download, alpha verify, fuse, register

### 1.0 — Verify LoRA alpha (HARD GATE)

`model/scripts/fuse_lora_ckpt.py:5` hardcodes `ALPHA = 32`. If the ETHRC
posttraining used a different alpha, the fused weights are silently wrong.

```bash
aws s3 cp \
  s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-libero/outputs/posttraining/video2world_lora/2b_libero_cosmos/config.yaml \
  /tmp/ethrc_unified_config.yaml

grep -i 'lora_alpha\|alpha' /tmp/ethrc_unified_config.yaml
```

- Expected: `lora_alpha: 32`. **If 32, proceed.**
- If anything else (e.g., `lora_alpha: 64`), edit `fuse_lora_ckpt.py:5`
  to match the actual alpha **before** running fusion. Save & document.
- Also note `lora_rank` from the same file — sanity check after fusion that
  the rank inferred matches.

### 1.1 — Download + fuse + place

```bash
cd /workspace/mimic-video/model
. scripts/env_setup.sh

mkdir -p /tmp/cosmos_dl
aws s3 cp \
  s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-libero/outputs/posttraining/video2world_lora/2b_libero_cosmos/checkpoints/model/iter_000007000.pt \
  /tmp/cosmos_dl/iter_000007000.pt

python scripts/fuse_lora_ckpt.py /tmp/cosmos_dl/iter_000007000.pt
# → /tmp/cosmos_dl/iter_000007000_fused.pt   (expect ~4 GB)

mv /tmp/cosmos_dl/iter_000007000_fused.pt \
   checkpoints/video_backbone/v2w_libero_cosmos_unified_iter_000007000_fused.pt
rm /tmp/cosmos_dl/iter_000007000.pt
```

Acceptance: the fused file is ~4 GB (matches upstream `_fused.pt` sizes).
If it's the same size as input (~12 GB), the fuse silently no-op'd → bug.

### 1.2 — Register in `world2action_model.py`

Edit `model/cosmos_predict2/configs/defaults/world2action_model.py:61`:

```python
VIDEO_MODEL_CKPT_NAMES = [
    "v2w_pretrained_cosmos",
    "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused",
    "v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused",
    "v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused",
    "v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused",
    "v2w_libero_cosmos_unified_iter_000007000_fused",   # NEW (ETHRC unified)
]
```

### 1.3 — Remove the `save_iter=99999999` hardcode (review 1A)

Edit `model/cosmos_predict2/configs/experiment/world2action.py:104-106`:

```python
# BEFORE
if "libero" in data_config:
    cfg["checkpoint"]["save_iter"] = 99999999
    cfg["trainer"]["run_validation"] = False

# AFTER
if "libero" in data_config:
    # save_iter follows BASE default (1000); override per-run via CLI.
    # Validation stays off — full val pass at iter 0 takes days otherwise.
    cfg["trainer"]["run_validation"] = False
```

This makes the CLI override no longer load-bearing. Belt + suspenders for
the next hard gate (1.5).

### 1.4 — Sanity check that the experiment exists

```bash
python -c "
from cosmos_predict2.configs.defaults.data_action import DATA_CONFIGS
assert 'libero_object_full' in DATA_CONFIGS, 'data_config not auto-discovered'
print('ok')
"

python -m scripts.train --config=cosmos_predict2/configs/config.py \
  -- experiment=w2a_libero_object_full_v2w_libero_cosmos_unified_iter_000007000_fused_lr1.000e-04_layer20_bsz128 \
  trainer.max_iter=2 trainer.run_validation=False \
  dataloader_train.batch_size=1 \
  --print-config 2>&1 | grep -E 'video_dit_path|save_iter'
```

Acceptance:
- DATA_CONFIGS contains `libero_object_full` (it's auto-discovered from
  `dataloading/*.yaml`, but worth confirming).
- `video_dit_path` resolves to the absolute path of the fused file from 1.1.
- `save_iter` does not appear as 99999999.

### 1.5 — Save-iter smoke verification (HARD GATE)

```bash
torchrun --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment=<EXP> \
  trainer.max_iter=200 \
  trainer.run_validation=False \
  trainer.logging_iter=50 \
  checkpoint.save_iter=100 \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=2

# Acceptance: at least 2 .pt files appear in
#   checkpoints/vam/libero/<EXP>/{iter_000000100,iter_000000200}/...
# If zero checkpoints land, the override+edit isn't working — STOP.
```

---

## Step 2 — Data prep (S3 HDF5s → zarrs) — runs LOCALLY on the 5090

Data prep is CPU-bound (mostly zarr writes + T5 forward pass). T5-11B is
already on the 5090 box, so no reason to do this on AWS. Final zarrs ship
to AWS for training.

### 2.1 — Download the libero_object subset of the regenerated HDF5s

```bash
cd /home/ethrc/Desktop/mimic-video
mkdir -p data/libero_object_regen
aws s3 cp \
  s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-libero/datasets/libero_cosmos/all_episodes/ \
  data/libero_object_regen/ \
  --recursive \
  --exclude "*" \
  --include "*suite=libero_object*"

# Expect ~500 .hdf5 files, ~3 GB total (object subset).
ls data/libero_object_regen/ | wc -l    # → ~500
du -sh data/libero_object_regen/         # → ~3G
```

### 2.2 — `process_libero.py` → zarrs

```bash
. model/scripts/env_setup.sh
cd data_preprocessing/action/
python process_libero.py \
  --input-dir ../../data/libero_object_regen/ \
  --output-dir ../../data/libero_object_full/
```

Output: zarrs under `data/libero_object_full/episode_NNNN.zarr`, expected
total ~5–15 GB (LIBERO is 256-px; lower than bi_yams).

### 2.3 — `precompute_t5.py`

```bash
python precompute_t5.py --dataset-path ../../data/libero_object_full
```

Uses the T5-11B already at `model/checkpoints/text_encoder/t5-11b/`. Per
the bi_yams notes, the apex-FusedRMSNorm patch in this script is a no-op
on sm_120 — leave it.

### 2.4 — Ship zarrs to AWS

Once Steps 2.2-2.3 are clean, rsync `data/libero_object_full/` to the AWS
instance:

```bash
# From the 5090, after AWS instance is up:
rsync -avzh --progress data/libero_object_full/ \
  ec2-user@<aws-host>:/workspace/data/libero_object_full/
```

~5–15 GB transfer, ~1–2 hours on a typical home upstream. Or push to S3 first
and `aws s3 cp` from AWS — faster on a fat pipe.

### 2.5 — Dataset acceptance check (HARD GATE) — runs locally on 5090

Before shipping to AWS, sanity-check on the 5090 itself:

```bash
cd /home/ethrc/Desktop/mimic-video/model
python - <<'PY'
from omegaconf import OmegaConf
from hydra.utils import instantiate
from cosmos_predict2.configs.defaults import data_action
data_action.register_training_and_val_action_data()
cfg = OmegaConf.load('cosmos_predict2/configs/dataloading/libero_object_full.yaml')
cfg.dataset.dataset.data_dir = '../data/libero_object_full'
ds = instantiate(cfg.dataset.dataset)
item = ds[0]
print('keys:', list(item.keys()))
print('obs/workspace_rgb:', item['obs']['workspace_rgb'].shape)         # expect [5, 3, 256, 256] or similar
print('action lowdim concat:', item['action']['lowdim_concat'].shape)   # expect [60, 10]
print('language_embedding:', item['obs']['language_embedding'].shape)
PY
```

Acceptance:
- `obs/workspace_rgb` dtype float32, shape `[5, 3, H, W]`.
- `action/lowdim_concat` shape `[60, 10]` (60-step horizon, 10-d action).
- `obs/language_embedding` present and non-zero.

If anything's off, **STOP** — dataloader can't be wrong before launching a
50K-iter run.

---

## Step 3 — Config edit (1 path string)

Edit `model/cosmos_predict2/configs/dataloading/libero_object_full.yaml`:

```yaml
defaults:
  - dataset: libero
  - policy_io: libero
  - _self_

dataset:
  dataset:
    data_dir: /workspace/data/libero_object_full   # ← from Step 2.2 output
```

That's it. `dataset/libero.yaml` and `policy_io/libero.yaml` are already wired.

---

## Step 4 — AWS env setup

### Instance

8× A100 80 GB. The standard option is **p4d.24xlarge** (us-east-1) or
**p4de.24xlarge** for 80 GB cards. Ubuntu DLAMI or NGC PyTorch container —
both ship apex/transformer_engine prebuilds for sm_80, which means **none
of the sm_120 5090 workarounds apply on AWS**. Specifically drop:

- `optimizer=adamw` override → use default `fusedadamw`
- `NVTE_FUSED_ATTN=0`
- venv-path `CUDA_HOME` override
- `libcudart.so` symlink
- T5LayerNorm apex monkey-patch (already a no-op on A100)

### Two venvs (review note from outside voice)

LIBERO benchmark deps (`mujoco`, old `gym`, `robosuite`) conflict with the
cosmos training stack. **Install LIBERO in a separate venv from training**:

```bash
# venv 1: training
cd /workspace/mimic-video/model
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra cu126
source .venv/bin/activate
python scripts/test_environment.py    # sanity

# venv 2: eval (separate dir)
deactivate
cd /workspace/mimic-video/eval/libero
uv venv .venv-eval
source .venv-eval/bin/activate
uv pip install -r LIBERO/requirements.txt
uv pip install -e LIBERO
# also need cosmos_predict2 importable for run.py — install model/ deps minus the
# heavy training-only stuff:
uv pip install -e ../../model
```

Single-venv installs are tempting but bite hard at the eval stage when
`mujoco` upgrades break `transformer_engine`.

### Disk budget

Data prep moved to the 5090 (Step 2). AWS only needs the final zarrs +
checkpoints, not raw HDF5s, T5-11B, or HF datasets for training.

| item | size |
|---|---|
| repo + venvs (training + eval) | ~12 GB |
| Wan tokenizer | ~1 GB |
| `v2w_pretrained_cosmos.pt` (only if needed) | ~5 GB |
| ETHRC fused backbone | ~4 GB |
| Upstream object_full backbone + decoder (for baseline eval, rsync from 5090) | ~7 GB |
| `libero_object_full` zarrs (rsync from 5090) | ~5–15 GB |
| LIBERO benchmark + HF datasets (needed for sim eval at runtime) | ~15 GB |
| Action decoder checkpoints (50K iters, save_iter=5000 → 10 saves × ~3 GB) | ~30 GB |

**Total: ~80–100 GB**, comfortable on a p4d.24xlarge's 8 TB instance store.

T5-11B stays on the 5090; not needed on AWS since precompute happened locally
in Step 2.3.

---

## Step 5 — Launch training

```bash
cd /workspace/mimic-video/model
source .venv/bin/activate

EXP=w2a_libero_object_full_v2w_libero_cosmos_unified_iter_000007000_fused_lr1.000e-04_layer20_bsz128

# WandB integration — same pattern as bi_yams (~/.netrc with the team API key).
# job.group is auto-set to "libero" by the world2action_pipe registration.
# job.name is set to EXP automatically; tag via WANDB_TAGS env var if desired.
export WANDB_PROJECT=vam
export WANDB_TAGS=libero,libero_object,ethrc_unified_backbone

TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=8 --master_port=12341 \
  -m scripts.train --config=cosmos_predict2/configs/config.py \
  -- experiment="$EXP" \
  trainer.max_iter=50000 \
  trainer.run_validation=False \
  trainer.logging_iter=100 \
  checkpoint.save_iter=5000 \
  dataloader_train.num_workers=12 \
  dataloader_train.prefetch_factor=4
```

### Notes

- `bsz=128 global` on 8 GPUs → micro=16/GPU. A100 80 GB has plenty of headroom.
- `max_iter=50000` matches upstream eval.sh checkpoint iters (50,022 / 50,274
  / 51,212 across goal/object/spatial). 80K was a guess from the bi_yams
  plan; libero converges earlier.
- LIBERO dataloading is uniform per-trajectory; expect smoother loss curves
  than bi_yams.
- Per-iter time on 8× A100 80 GB at micro=16: ≈ 0.6–0.9 s expected. 50K iters
  → 9–13 h.

### What we expect to see

- Loss curve: starts in the 30s (backbone has seen libero_object frames in
  cosmos finetune), drops fast.
- WandB run lives under `vam` project, group=`libero`, name=`<EXP>`.
- 10 checkpoints under `checkpoints/vam/libero/<EXP>/iter_NNNN/...`.

---

## Step 6 — Sim eval (ETHRC + upstream baseline A/B)

The plan keeps `object_full` (upstream) and adds a parallel `object_full_ethrc`
entry. eval.sh runs both; we get a direct A/B.

### Concrete `eval/libero/eval.sh` diff

Add a new associative array after the existing `object_full=(...)` block
(around line 67):

```bash
declare -A object_full_ethrc=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_libero_object_full_v2w_libero_cosmos_unified_iter_000007000_fused_lr1.000e-04_layer20_bsz128
  [action_model]=${checkpoint_dir}/action_decoder/w2a_libero_object_full_v2w_libero_cosmos_unified_iter_000007000_fused_lr1.000e-04_layer20_bsz128_iter_000050000.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_libero_cosmos_unified_iter_000007000_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/libero_object_full.json
  [suite]=libero_object
)
```

(`iter_000050000.pt` = whatever the highest saved iter from Step 5 is — adjust
once training finishes; commonly the last save.)

Then change line 124:

```bash
# Before:
models=(goal_half goal_tenth goal_one object_full object_half object_tenth object_one spatial_full spatial_tenth spatial_one)
# After:
models=(object_full object_full_ethrc)
```

Set `checkpoint_dir` (line 29) to the AWS path where checkpoints landed.
Copy or rsync the upstream files (`v2w_libero_object_agentview_*_fused.pt`
and the upstream `iter_000050274.pt` action decoder) from the local 5090 box
to the AWS instance at `/workspace/mimic-video/model/checkpoints/...` first.

### Run

```bash
cd /workspace/mimic-video/eval/libero
source .venv-eval/bin/activate          # the eval-only venv
bash eval.sh
```

### Expected output + budget

- Per model variant: 36 stop_steps × 5 ranks × 50 trials/task × 10 tasks = 90,000
  rollouts. Two variants (upstream + ETHRC) = 180,000 rollouts total.
- On 8× A100, 2-per-GPU = 16 concurrent. **Expect ~6–10 hours wall-clock.**
  (The earlier 2–4 h estimate was for one variant; A/B doubles it.)
- `results/` will contain per-task success rates for both `object_full` and
  `object_full_ethrc`. Compare aggregate.

### What constitutes success

- ETHRC ≥ upstream on aggregate libero_object success rate → finetune
  helped.
- ETHRC ~= upstream → finetune was neutral; the unified backbone lacks a
  per-suite specialization advantage, but isn't worse.
- ETHRC << upstream → likely cause is visual distribution drift (Risk #1
  below). Investigate before second training run.

---

## Risks (still open)

These are real risks the steps above don't fully neutralize:

1. **Visual distribution match**. The S3 HDF5s are matched to the cosmos
   finetune training distribution, but `process_libero.py` may apply its
   own rotation/crop/resize on top — diverging from what the backbone saw.
   **Mitigation**: spot-check a frame from an output zarr against a reference
   frame from the cosmos posttraining mp4s
   (`s3://…/datasets/libero_cosmos_mp4/train/videos/`) before launching
   training. ~10 min.

2. **`iter_000007000` may not be the best ETHRC checkpoint**. WandB run
   <https://wandb.ai/eth-robotics-club/posttraining/runs/j0hjrgb6> shows
   val loss / sample quality across iters. **Mitigation**: skim the run
   before fusing. If val loss bottoms at iter 5500, fuse that one instead
   (the S3 checkpoints folder has all iters from 1500 onwards).

3. **A100 80 GB availability in eu-central-1**. AWS account is configured
   for eu-central-1; p4d.24xlarge (8× A100 40 GB) and p4de.24xlarge (80 GB)
   both have sporadic capacity. **Mitigation**: try us-east-1 if eu blocks,
   accept ~$0.05/GB cross-region S3 transfer.

4. **eval.sh runtime ambiguity**. The 5090 box runs Step 0 with capped
   `stop_steps`. The AWS Step 6 uses the full sweep. If the local smoke
   test runtime is ~30 min/task with caps, full sweep on 8 GPUs is
   ~6–10 h — but if any task hangs (LIBERO has known sapien-render race
   conditions), eval can deadlock. **Mitigation**: time-cap eval at 12h
   and inspect.

5. **Multi-node not in scope.** All Step 5 throughput estimates assume
   single-node 8× A100. If the AWS allocation is 4× A100, double the
   wall-clock and reduce micro-batch.

---

## Compute / wall-clock budget

| step | est. time | venue |
|---|---|---|
| Step 0 — local upstream eval (capped) | ~1–2 h | 5090 |
| Step 2.1 — S3 HDF5 download | ~5 min | 5090 |
| Step 2.2 — process_libero (zarr conversion) | ~15 min | 5090 |
| Step 2.3 — precompute_t5 | ~5 min | 5090 |
| Step 2.4 — rsync zarrs to AWS (5–15 GB) | ~30–90 min | 5090 → AWS |
| Step 2.5 — dataset acceptance check | 5 min | 5090 (before rsync) |
| Step 1.0 — alpha verify | 5 min | AWS |
| Step 1.1 — download + fuse + place ETHRC backbone | ~10 min | AWS |
| Step 1.2-1.3 — code edits | 5 min | AWS |
| Step 1.4-1.5 — sanity + save_iter smoke | ~30 min | AWS |
| Step 5 — train 50K iters @ 8× A100 80 GB | ~9–13 h | AWS |
| Step 6 — sim eval A/B (object_full + object_full_ethrc) | ~6–10 h | AWS |

**End-to-end on AWS: ~16–24 h once data and backbone are in place.**
**Local 5090 work: ~3–5 h** (Step 0 + data prep + rsync).

---

## NOT in scope (explicitly deferred)

- **Per-suite ablations** (spatial, goal, libero_10) — user picked object only.
- **Non-finetuned cosmos baseline** (`v2w_pretrained_cosmos`) — out of scope; ETHRC
  vs upstream A/B is sufficient calibration. (Pretrained-cosmos action decoder
  for libero would need its own ~16h training run.)
- **Real-robot data mixing** (carton-box / towel) — sim eval is in-distribution
  to LIBERO; mixing real-robot data hurts.
- **Cosmos backbone re-finetuning** — using iter_7000 as-is. If wandb says a
  different iter is better, fuse that instead (Risk #2).
- **Hyperparameter sweeps** — single point in the experiment grid (lr 1e-4,
  layer 20, bsz 128). Sweep available later via `lrs`/`bszs` lists in
  `world2action.py`.
- **The official mimic-video data prep pipeline** (LIBERO install +
  `download_libero_datasets` + `regenerate_libero` + `process_libero`) —
  superseded by direct S3 HDF5 download. Skipping saves ~3-4 h of friend's
  compute and guarantees visual match with the backbone.
- **Multi-node training** — single-node 8× A100 only. Plan does not handle
  NCCL/IB cross-node.

---

## Asks of the user (TBD)

- Specific AWS instance type (p4d.24xlarge / p4de.24xlarge / other)?
- Once on AWS, where will zarrs land? (Default in plan: `/workspace/data/libero_object_full/`.
  Goes into `libero_object_full.yaml:data_dir` after rsync from the 5090.)
- Does the wandb run support iter_7000 as the best ETHRC checkpoint, or should we
  use an earlier one?

(Friend coordination is no longer pending — steps 1+2 of the README are done
locally on the 5090, and Path B uses S3 HDF5s instead of the HF demos.)

---

## What already exists (reused, no new code)

| asset | path |
|---|---|
| Data prep — `process_libero.py` | `data_preprocessing/action/process_libero.py` |
| Data prep — `precompute_t5.py` | `data_preprocessing/action/precompute_t5.py` |
| LoRA fuse script | `model/scripts/fuse_lora_ckpt.py` |
| libero pipe (10-d action, 60 horizon) | `model/cosmos_predict2/configs/defaults/world2action_pipe.py:10` |
| libero dataset config | `model/cosmos_predict2/configs/dataloading/dataset/libero.yaml` |
| libero policy_io config | `model/cosmos_predict2/configs/dataloading/policy_io/libero.yaml` |
| libero_object_full top-level dataloading config | `model/cosmos_predict2/configs/dataloading/libero_object_full.yaml` |
| Auto-experiment generator | `model/cosmos_predict2/configs/experiment/world2action.py` |
| Sim eval harness | `eval/libero/eval.sh`, `eval/libero/run.py` |
| Upstream backbone (used at Step 0 + as baseline) | `model/checkpoints/video_backbone/v2w_libero_object_agentview_..._fused.pt` |
| Upstream object_full action decoder (calibration anchor) | `model/checkpoints/action_decoder/w2a_libero_object_full_v2w_libero_object_agentview_..._iter_000050274.pt` |

**New code total: 1 line added (`world2action_model.py:67`), 1 hardcode removed
(`world2action.py:105`), 1 path edit (`libero_object_full.yaml:9`), 1 new
associative array in `eval.sh`.** All other work is data movement and
verification.

---

## Failure modes (per-step, by review)

| step | failure mode | covered by |
|---|---|---|
| Step 0 | eval.sh import error / LIBERO sapien-render fails | Step 0 IS the test for this |
| Step 1.0 | `lora_alpha != 32` → silently wrong fused weights | Hard gate: read S3 config.yaml |
| Step 1.1 | fuse runs but produces same-size file (no-op) | Acceptance: file size ~4 GB not 12 GB |
| Step 1.5 | `save_iter` override broken → 50K-iter run with 0 saves | Hard gate: smoke test 200 iters with save_iter=100 |
| Step 2.4 | dataloader item shape mismatch → silent training divergence | Hard gate: print shapes before training |
| Step 5 | NCCL hang on a flaky node | `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200` set |
| Step 6 | LIBERO sim deadlock | 12-hour wall-clock cap |

No failure modes are silent + uncovered. ✓

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | (codex unavailable) |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR | 6 issues (4 P0/P1, 2 P2/P3); 2 cross-model tensions, all resolved |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | n/a (training run) |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | n/a (research run) |

- **OUTSIDE VOICE:** Claude subagent challenge ran. Found 6 things the eng review
  missed/underweighted; all 6 surfaced and resolved (3 patched into the plan as
  hard gates, 2 promoted to user-decision tensions and approved, 1 was a
  reflection of the locked decisions not yet patched into the plan body —
  patched here).
- **CROSS-MODEL CONSENSUS:** Both reviewers agree on save_iter hazard, max_iter
  overshoot, eval.sh hand-wave, and alpha verification gate. Outside voice
  added the visual-distribution-as-primary insight (T-1) and local-Step-0
  smoke test (T-2). User approved both.
- **UNRESOLVED:** 0
- **VERDICT:** ENG CLEARED — ready to implement. Begin with Step 0 (local
  5090 upstream eval) today; Step 1 onward gated on Step 0 success and
  friend's S3 HDF5 download.
