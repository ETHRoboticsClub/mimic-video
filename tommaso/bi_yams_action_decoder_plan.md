# bi_yams (real-robot) Action Decoder Training Plan — ETHRC YAMS-finetuned Cosmos, AWS 8× A100 80 GB

Train the real-robot action decoder on top of the ETHRC YAMS-finetuned Cosmos
backbone, using the EXACT trajectories the cosmos was finetuned on
(`ETHRC/robot-learning-fs26` on Hugging Face). Real-robot deployment for eval,
not sim.

---

## TL;DR

- **Backbone**: ETHRC YAMS-finetuned Cosmos,
  `s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-yam/outputs/posttraining/video2world_lora/2b_yams/checkpoints/model/iter_000007000.pt`.
  Confirmed `lora_alpha=16, lora_rank=16`. 11.9 GB raw → fuse → filter to net.* → **~4 GB** fused.
- **Data**: HF `ETHRC/robot-learning-fs26` — verified byte-identical to the
  cosmos posttraining input at `s3://.../yams_lerobot/`. **244 episodes,
  153K frames @ 30 fps, 2 tasks** (carton-box + towel), 14-dim joint action,
  3 camera views (topdown, wrists). lerobot v3.0 format. **324 MB total.**
- **Hardware**: AWS **8× A100 80 GB** (new p4de.24xlarge — separate from the
  running libero p4d 40 GB instance). `bsz=128 global` → micro=16/GPU, **no
  grad_accum**.
- **Configs**: `bi_yams.yaml`, `dataset/bi_yams.yaml`, `policy_io/bi_yams.yaml`,
  and the `bi_yams` decoder pipe entry already exist from the prior 5090
  attempt. **Zero new configs.**
- **Data prep script**: `process_bi_yams.py` already reads lerobot v3 (parquet
  + mp4). **Zero new data code.**
- **Training**: **10,000 iters** at expected ~38 s/iter = **~4 days
  wall-clock**. Saves every 1000 iters → 10 checkpoints.
- **Eval**: real-robot deployment. User owns `mimic_adapter.py` rewrite; I
  deliver the trained checkpoint and an interface note.

---

## Why this should be smoother than libero

| | libero (current AWS run) | bi_yams (this plan) |
|---|---|---|
| GPU memory | 40 GB → micro=4 + grad_accum=4 | **80 GB** → micro=16, no grad_accum |
| Decoder `max_horizon` | 61 | **31** (~2× cheaper per forward) |
| Per-iter wall-clock | ~77 s | **~38 s** (2× faster) |
| 10K iters wall-clock | ~9 days | **~4 days** |
| Configs | Already wired | Already wired |
| Data converter | Custom `process_libero_s3.py` (new) | `process_bi_yams.py` (already exists) |
| Dataset prep on AWS | 21 GB rsync | 324 MB download from HF directly |

---

## Decisions locked in

| Knob | Choice | Source |
|---|---|---|
| Suite | `bi_yams` (real-robot) | user |
| Data | HF `ETHRC/robot-learning-fs26` | user |
| Task scope | **Both tasks** (carton-box + towel) | user 2026-05-11 |
| Backbone | ETHRC `2b_yams/iter_000007000.pt` | user |
| `lora_alpha` for fuse | `16` (verified from S3 config.yaml) | repo |
| Hardware | New AWS p4de.24xlarge (8× A100 80 GB) | user 2026-05-11 |
| `max_iter` | **10,000** (~4 days first pass) | user 2026-05-11 |
| `save_iter` | **200** (50 checkpoints, ~2h apart) | user 2026-05-11 |
| Smoke test | **50 iters single-GPU, save_iter=25** (fail fast on plumbing) | user 2026-05-11 |
| Validation during training | disabled (libero block doesn't apply) | n/a |
| Optimizer | `fusedadamw`, lr ≈ 1e-4 | repo default |
| Cross-attention layer | `xattn_layer_idx=20` | repo default |
| Global batch | 128 | repo default |
| Real-robot eval ownership | **User**, with my deliverable: checkpoint + interface notes | user 2026-05-11 |

---

## Inputs

### Already on the local 5090 (no action)

```
model/checkpoints/text_encoder/t5-11b/                                                                    # 45 GB
model/checkpoints/video_backbone/tokenizer/tokenizer.pth                                                  # Wan VAE

model/cosmos_predict2/configs/dataloading/bi_yams.yaml                                                    # top-level
model/cosmos_predict2/configs/dataloading/dataset/bi_yams.yaml
model/cosmos_predict2/configs/dataloading/policy_io/bi_yams.yaml
model/cosmos_predict2/configs/defaults/world2action_pipe.py:40                                            # bi_yams decoder pipe
                                                                                                          # max_horizon=31, in_channels=14, out_channels=14

data_preprocessing/action/process_bi_yams.py                                                              # lerobot v3 → zarr
data_preprocessing/action/precompute_t5.py
model/scripts/fuse_lora_ckpt.py                                                                           # patched (--alpha CLI) during libero plan
```

### To download

**On the 5090** (data prep happens here, AWS gets only zarrs):

1. **HF dataset** `ETHRC/robot-learning-fs26` (324 MB, ~1 min on a fast pipe):
   ```bash
   hf download --repo-type=dataset ETHRC/robot-learning-fs26 \
     --local-dir /home/ethrc/Desktop/mimic-video/data/robot-learning-fs26
   ```

**On the AWS instance** (training):

1. **ETHRC YAMS backbone** (~12 GB raw → 4 GB fused):
   ```
   s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-yam/outputs/posttraining/video2world_lora/2b_yams/checkpoints/model/iter_000007000.pt
   ```

2. **Config.yaml for alpha verification** (already pulled — confirmed
   `lora_alpha=16, lora_rank=16`). On AWS we re-pull just to confirm:
   ```
   s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-yam/outputs/posttraining/video2world_lora/2b_yams/config.yaml
   ```

### Skip / NOT needed

- LIBERO benchmark install (no sim eval for bi_yams)
- The OLD `data/bi_yams_carton/` zarrs from prior 5090 attempt (used a
  different dataset; discard or ignore)
- T5-11B on AWS (precompute on 5090, embeddings bake into zarrs)
- The libero AWS instance (independent — keep it running for that workload)

---

## Step 0 — Local upstream-baseline smoke test (SKIP)

For libero, Step 0 was running the upstream paper's already-trained
`object_full` decoder against sim. **For bi_yams there's no upstream-trained
decoder, no sim, and no calibration anchor.** Skip Step 0.

Risk: we have no "expected success rate" to compare against. The first
training output IS the first signal.

---

## Step 1 — Backbone download, alpha verify, fuse, register (on AWS)

### 1.0 — Verify LoRA alpha (HARD GATE, already done locally — re-verify on AWS)

```bash
mkdir -p /tmp/cosmos_dl
aws s3 cp \
  s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-yam/outputs/posttraining/video2world_lora/2b_yams/config.yaml \
  /tmp/cosmos_dl/yams_config.yaml
grep -E "lora_alpha|lora_rank" /tmp/cosmos_dl/yams_config.yaml
```

Expected: `lora_alpha: 16`, `lora_rank: 16`. **Already verified 2026-05-11.**

### 1.1 — Download + fuse + filter to net.* + place

```bash
cd /home/ubuntu/workspace/mimic-video/model
source .venv/bin/activate

aws s3 cp \
  s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-yam/outputs/posttraining/video2world_lora/2b_yams/checkpoints/model/iter_000007000.pt \
  /tmp/cosmos_dl/yams_iter_000007000.pt

python scripts/fuse_lora_ckpt.py /tmp/cosmos_dl/yams_iter_000007000.pt --alpha 16

# Strip EMA keys to match upstream fused format (same as libero plan):
python - <<'PY'
import torch
src = "/tmp/cosmos_dl/yams_iter_000007000_fused.pt"
dst = "/tmp/cosmos_dl/v2w_yams_unified_iter_000007000_fused.pt"
ckpt = torch.load(src, weights_only=False, map_location="cpu")
filtered = {k: v for k, v in ckpt.items() if k.startswith("net.")}
print(f"{len(ckpt)} -> {len(filtered)} keys")
torch.save(filtered, dst)
PY

mv /tmp/cosmos_dl/v2w_yams_unified_iter_000007000_fused.pt \
   checkpoints/video_backbone/v2w_yams_unified_iter_000007000_fused.pt
rm /tmp/cosmos_dl/yams_iter_000007000.pt /tmp/cosmos_dl/yams_iter_000007000_fused.pt
ls -lh checkpoints/video_backbone/v2w_yams_unified_iter_000007000_fused.pt
# Expect ~4 GB
```

### 1.2 — Register backbone in `world2action_model.py:61`

Add one line:

```python
VIDEO_MODEL_CKPT_NAMES = [
    "v2w_pretrained_cosmos",
    "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused",
    "v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused",
    "v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused",
    "v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused",
    "v2w_libero_cosmos_unified_iter_000007000_fused",
    "v2w_yams_unified_iter_000007000_fused",   # NEW
]
```

### 1.3 — Sanity check experiment registration

```bash
python -c "
from cosmos_predict2.configs.defaults.data_action import DATA_CONFIGS
assert 'bi_yams' in DATA_CONFIGS, sorted(DATA_CONFIGS.keys())
print('bi_yams data config registered')
"

EXP=w2a_bi_yams_v2w_yams_unified_iter_000007000_fused_lr1.000e-04_layer20_bsz128
python -m scripts.train --config=cosmos_predict2/configs/config.py \
  -- experiment="$EXP" \
  trainer.max_iter=2 trainer.run_validation=False \
  dataloader_train.batch_size=1 \
  --print-config 2>&1 | grep -E "video_dit_path|save_iter|in_channels|max_horizon"
```

Acceptance:
- `video_dit_path` resolves to the fused YAMS backbone
- `max_horizon=31`, `in_channels=14`, `out_channels=14`
- `save_iter` is BASE default (1000), NOT 99999999 (libero hardcode doesn't apply for bi_yams)

### 1.4 — 50-iter single-GPU smoke test (HARD GATE)

Fast smoke (~5 min, 1 GPU at micro=1) to catch plumbing issues before
committing to multi-GPU compute. Saves at iter 25 and 50:

```bash
torchrun --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment="$EXP" \
  trainer.max_iter=50 \
  trainer.run_validation=False \
  trainer.logging_iter=10 \
  checkpoint.save_iter=25 \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=2
```

Acceptance:
- 2 saves on disk at iter 25 and 50 (model + optim + scheduler + trainer)
- Loss finite and trending down across 50 iters (no NaN, no diverging)
- No CUDA OOM, no NCCL hang
- `checkpoints/dataset_statistics/bi_yams.json` exists (or equivalent name)
  after iter 0 — confirms the action normalization stats file gets created
  during training (needed for inference)

### 1.5 — 50-iter 8-GPU throughput probe (HARD GATE, added from review)

Measure real per-iter wall-clock at the actual training config BEFORE
committing to the 10K run.

```bash
# Clean smoke ckpts first so probe runs from iter 0
rm -rf checkpoints/vam/bi_yams/<EXP>/checkpoints/

ulimit -n 1048576
NVIDIA_LIBS=$(find $PWD/.venv/lib/python3.10/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}
export CUDA_HOME=$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 --master_port=12347 \
  -m scripts.train --config=cosmos_predict2/configs/config.py \
  -- experiment="$EXP" \
  trainer.max_iter=50 \
  trainer.run_validation=False \
  trainer.logging_iter=5 \
  checkpoint.save_iter=999999 \
  dataloader_train.num_workers=4 \
  dataloader_train.prefetch_factor=2
```

Acceptance:
- **No OOM at micro=16 on 80 GB cards.** If OOM, fall back to grad_accum
  (same approach we took on libero 40 GB cards).
- Per-iter steady-state time logged via `iter_speed` callback (fires
  every `logging_iter=5` iters).
- **Compute real ETA**: per_iter × 10000 / 3600 = hours for full run.
- If per-iter > 60s, reconsider max_iter (10K → 5K) before launching long run.

Clean ckpts before full launch:

```bash
rm -rf checkpoints/vam/bi_yams/<EXP>/checkpoints/
```

---

## Step 2 — Data prep (HF → zarrs, ALL ON AWS — review Issue 2)

Moved from 5090-then-rsync to AWS-direct. Saves ~3 hours of transfer time.
All steps run on the p4de.24xlarge instance.

### 2.1 — Download HF dataset on AWS

```bash
cd /home/ubuntu/workspace/mimic-video
source model/.venv/bin/activate
pip install -U "huggingface_hub[cli]"  # if not already present
hf download --repo-type=dataset ETHRC/robot-learning-fs26 \
  --local-dir data/robot-learning-fs26
du -sh data/robot-learning-fs26/   # expect ~324 MB
```

### 2.2 — process_bi_yams → zarrs (on AWS)

```bash
cd /home/ubuntu/workspace/mimic-video
source model/.venv/bin/activate
cd data_preprocessing/action/
python process_bi_yams.py \
  --dataset-dir ../../data/robot-learning-fs26 \
  --output-dir   ../../data/yams_fs26 \
  --num-workers 16 --overwrite   # AWS has plenty of CPU; bump workers
```

Output: 244 per-episode `episode_NNNN.zarr` files.

Per the prior 5090 attempt on a similar dataset (136 eps, 99K frames), expect
~1.1 GB per zarr. **244 eps × 1.1 GB ≈ ~270 GB of zarrs**.

⚠️ **That's a lot.** Need to verify 5090 disk free space (~377 GB last
checked) AND AWS instance disk (290 GB on the previous one, may need bigger
EBS for p4de). Will check before launching.

### 2.3 — precompute_t5 (on AWS — needs T5-11B)

T5-11B isn't on the AWS instance by default. Download via HF auto-cache on first
invocation (45 GB, ~5-10 min on AWS pipe). The script handles multi-task
correctly when `--prompt` is NOT passed (each zarr's `language_instruction` is
read individually).

```bash
# precompute_t5.py uses T5_MODEL_DIR from imaginaire.constants. May need to
# point it at the HF cache, or download T5-11B explicitly first:
hf download google-t5/t5-11b \
  --local-dir /home/ubuntu/workspace/mimic-video/model/checkpoints/text_encoder/t5-11b

python precompute_t5.py --dataset-path ../../data/yams_fs26
```

Two unique task strings → two T5 embeddings → each broadcast to its own
episodes. ~30s on a single A100.

After this completes, T5-11B can be deleted to free 45 GB:
```bash
rm -rf /home/ubuntu/workspace/mimic-video/model/checkpoints/text_encoder/t5-11b
```

### 2.4 — Dataset acceptance check (HARD GATE, on AWS)

```bash
cd /home/ubuntu/workspace/mimic-video/model
source .venv/bin/activate
python - <<'PY'
import zarr, glob, numpy as np
zs = sorted(glob.glob('../data/yams_fs26/*.zarr'))
assert len(zs) == 244, f"expected 244 zarrs, got {len(zs)}"
print(f'{len(zs)} zarrs OK')

z0 = zarr.open(zs[0], mode='r')
print('keys:', sorted(z0.keys()))
print('workspace_rgb:', z0['workspace_rgb'].shape, z0['workspace_rgb'].dtype)
print('joint_state_lowdim:', z0['joint_state_lowdim'].shape)
print('joint_action_lowdim:', z0['joint_action_lowdim'].shape)
print('language_embedding:', z0['language_embedding'].shape)

# CRITICAL: verify both task strings appear AND have DISTINCT embeddings
# (catches a bug where multi-task gets broadcast as a single embedding)
prompts_and_embeds = {}
for z in zs:
    zg = zarr.open(z, mode='r')
    p = bytes(zg['language_instruction'][0]).decode()
    e_mean = float(np.abs(zg['language_embedding'][:]).mean())
    if p not in prompts_and_embeds:
        prompts_and_embeds[p] = e_mean
    elif abs(prompts_and_embeds[p] - e_mean) > 1e-6:
        print(f"WARN: prompt {p!r} has inconsistent embedding across episodes")

assert len(prompts_and_embeds) == 2, f"expected 2 unique tasks, got {len(prompts_and_embeds)}: {list(prompts_and_embeds)}"
mean_a, mean_b = list(prompts_and_embeds.values())
assert abs(mean_a - mean_b) > 1e-4, "Both task embeddings have identical mean — they may be identical (bug)"
print(f'2 distinct task embeddings confirmed: {list(prompts_and_embeds.keys())}')
PY
```

Acceptance:
- 244 zarrs total
- `workspace_rgb` shape `[T, 480, 640, 3] uint8` (or whatever native frame size is)
- `joint_action_lowdim` shape `[T, 14] float32`
- `language_embedding` present, non-zero
- **Both task strings appear AND have DISTINCT embeddings** (multi-task verified)

### 2.5 — (REMOVED — data is born on AWS, no rsync needed)

Per review Issue 2, all data prep happens on AWS. Zarrs sit at
`/home/ubuntu/workspace/mimic-video/data/yams_fs26/` ready for training.

---

## Step 3 — Config edit (1 path)

Edit `model/cosmos_predict2/configs/dataloading/bi_yams.yaml:8`:

```yaml
dataset:
  dataset:
    data_dir: /home/ubuntu/workspace/mimic-video/data/yams_fs26
```

---

## Step 4 — AWS env setup (new p4de.24xlarge, 80 GB cards)

```bash
# Spin up p4de.24xlarge instance (us-east-1 or eu-central-1)
# Same setup as the libero instance, just bigger cards.

git clone https://github.com/ETHRoboticsClub/mimic-video.git /home/ubuntu/workspace/mimic-video
cd /home/ubuntu/workspace/mimic-video/model
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra cu126
source .venv/bin/activate
python -c "import cosmos_predict2; print('cosmos ok')"

# Apply the same one-line edits we made for libero:
# - data_preprocessing/action/process_libero_s3.py (irrelevant here but harmless)
# - world2action_model.py: register v2w_yams_unified_iter_000007000_fused
# - world2action.py: save_iter=99999999 hardcode (libero block only) — already removed
# - fuse_lora_ckpt.py: --alpha CLI flag

# rsync ~/.aws/ from 5090 if S3 creds aren't available via IAM role
```

### Disk budget on AWS (p4de.24xlarge)

| item | size |
|---|---|
| repo + venv | ~12 GB |
| Wan tokenizer | ~1 GB |
| ETHRC YAMS fused backbone | ~4 GB |
| `yams_fs26` zarrs | **~270 GB** |
| Action decoder checkpoints (10K iters / 200 = 50 saves × ~3 GB) | ~150 GB |

**Total: ~440 GB.** Default p4de EBS is often 1 TB, but **verify the instance
type's local NVMe + EBS** before kicking off rsync. If EBS is small (~290 GB
like the libero instance had), enlarge to **≥500 GB** before training, OR plan
to prune old checkpoints periodically.

---

## Step 5 — Launch full training

```bash
SESSION=yams_train_$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s "$SESSION" -c /home/ubuntu/workspace/mimic-video/model bash

tmux send-keys -t "$SESSION" "
ulimit -n 1048576
source .venv/bin/activate
NVIDIA_LIBS=\$(find \$PWD/.venv/lib/python3.10/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=\${NVIDIA_LIBS}\${LD_LIBRARY_PATH:-}
export CUDA_HOME=\$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=vam
export WANDB_TAGS=bi_yams,real_robot,ethrc_yams_backbone,aws_8xa100_80gb,fs26_both_tasks

EXP=w2a_bi_yams_v2w_yams_unified_iter_000007000_fused_lr1.000e-04_layer20_bsz128
LOGDIR=checkpoints/vam/bi_yams/\$EXP
mkdir -p \$LOGDIR

torchrun --nproc_per_node=8 --master_port=12347 \\
  -m scripts.train --config=cosmos_predict2/configs/config.py \\
  -- experiment=\$EXP \\
  trainer.max_iter=10000 \\
  trainer.run_validation=False \\
  trainer.logging_iter=100 \\
  checkpoint.save_iter=200 \\
  dataloader_train.num_workers=4 \\
  dataloader_train.prefetch_factor=2 \\
  optimizer.lr=1.0e-04 2>&1 | tee \$LOGDIR/train.log
" Enter

# Important from libero learnings:
# - ulimit -n 1048576 (avoid shared-memory worker crashes)
# - num_workers=4 (12 caused dataloader bus error on 5090; safer default)
# - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (reduce fragmentation)
```

### Expected throughput

| GPU config | per-iter | 10K iter ETA |
|---|---|---|
| 8× A100 80 GB, micro=16, no grad_accum | **~38 s/iter** | **~4 days** |

**No grad_accum needed** because micro=16 fits cleanly in 80 GB. This is the
single biggest difference vs the libero run.

### Save cadence

`save_iter=200` → checkpoints at iters 200, 400, ..., 10000.
**50 checkpoints, ~2 hours apart.** First usable model at **~2 hours from
launch** (much earlier than the 4-day full run).

Disk cost: 50 × ~3 GB = **~150 GB** in checkpoints. Adding to the 270 GB zarrs
+ 4 GB backbone + 12 GB venv = **~440 GB**. Provision ≥500 GB EBS to be safe.
If disk runs tight mid-training, prune older checkpoints manually
(`rm -rf checkpoints/.../iter_000000200.pt` etc.) — they're not needed for
resume, only the LATEST is needed for that.

---

## Step 6 — Real-robot eval (USER OWNS)

### My deliverable — INFERENCE BUNDLE (all of these are needed, not just the .pt)

When training produces a usable checkpoint, the FULL bundle for robot
inference is:

| File | Size | What | Where on AWS |
|---|---|---|---|
| `v2w_yams_unified_iter_000007000_fused.pt` | ~4 GB | Cosmos backbone (frozen at inference) | `checkpoints/video_backbone/` |
| `<EXP>_iter_NNNN.pt` | ~1 GB | Action decoder | `checkpoints/vam/bi_yams/<EXP>/checkpoints/model/` |
| `bi_yams.json` (dataset_statistics) | small | Action de-normalization stats | `checkpoints/dataset_statistics/` |
| `tokenizer.pth` | ~1 GB | Wan VAE (decodes video features) | `checkpoints/video_backbone/tokenizer/` |

⚠️ **The dataset_statistics file is CRITICAL.** The action decoder predicts
in normalized space (VARIANCE normalization per `policy_io/bi_yams.yaml:13`).
At inference, the stats file de-normalizes back to raw joint commands.
Without it, the robot gets garbage values.

### Interface notes (what I document for the user)

A short markdown documenting:
- The model loader call (same as libero, just different `experiment_name`
  and paths). Reference: `eval/libero/run.py:140-191` (`VAMInference.step`)
- Expected input shapes:
  - `workspace_rgb`: `[5, 3, 480, 640] uint8` at 5 Hz (last 5 topdown frames
    rescaled to 480x640)
  - `joint_state_lowdim`: `[1, 14] float32` (current 14-dim joint pose)
  - Language prompt: one of the two task strings ("Pick & Place and Closing a
    Box" or "Fold the towel")
- Expected output: 14-dim joint deltas, horizon=30 at 10 Hz, with VARIANCE
  normalization. Must de-normalize using `bi_yams.json` before sending to
  the robot.
- The two task strings need to be passed as prompts at eval time, matching
  the task the robot is meant to perform.

### User owns

- `mimic_adapter.py` rewrite in `mimic-yams/` (the inference adapter)
- Live robot session for eval
- Per-task success criterion (e.g., "carton lid touches box body within 30 s")

---

## Risks (open)

1. **Per-iter wall-clock estimate is extrapolated from libero, not measured.**
   38 s/iter is a projection; could be 30-60 s in reality. First few iters
   on AWS will give us the real number. **Mitigation**: launch + observe first
   100 iters before declaring 4-day ETA.

2. **244 episodes might converge before 10K iters.** With limited data and a
   relatively simple objective, the model might plateau at iter 3K-5K.
   **Mitigation**: save_iter=1000 means we see the trajectory and can stop
   early if loss flatlines.

3. **270 GB of zarrs is a lot for an EBS volume.** Default p4de EBS is 1 TB
   on most config templates, but the libero instance had only 290 GB.
   **Mitigation**: explicitly request ≥500 GB EBS when provisioning.

4. **`iter_000007000` may not be the best ETHRC YAMS checkpoint.** Earlier
   checkpoints (2.5K-6.5K) all exist on S3. Without a wandb run to consult,
   we default to 7K (latest).
   **Mitigation**: if real-robot eval shows weird visual artifacts in cosmos
   forwards, try fusing iter_005000 or earlier instead.

5. **No baseline / calibration anchor.** Unlike libero where we had upstream's
   trained decoder at 90% sim success, here we have nothing to compare against.
   The first real-robot trial IS the eval.
   **Mitigation**: define a clear success criterion BEFORE eval (per task),
   so we can call success/failure objectively rather than retrofitting.

---

## Compute / wall-clock budget

| step | est. time | venue |
|---|---|---|
| AWS p4de provision + uv sync + env setup | ~30 min | AWS |
| AWS: HF dataset download (324 MB) | ~1 min | AWS |
| AWS: T5-11B download (45 GB, one-time) | ~5-10 min | AWS |
| AWS: process_bi_yams → zarrs (244 eps, 16 workers) | ~15-30 min | AWS |
| AWS: precompute_t5 | ~30 s | AWS |
| AWS: dataset acceptance | ~5 min | AWS |
| AWS: alpha verify + backbone fuse + place | ~15 min | AWS |
| AWS: smoke test 1 (50 iters, 1 GPU) | ~5 min | AWS |
| AWS: throughput probe (50 iters, 8 GPUs) | ~30 min | AWS |
| AWS: train **10K iters** at micro=16 on 80 GB | **~4 days** (TBD by probe) | AWS |
| **First usable checkpoint (iter 200)** | **~2 hours after training launch** | AWS |
| Local: real-robot eval (per checkpoint) | 30-60 min per checkpoint | robot |

---

## NOT in scope (deferred)

- Wrist-camera CNN encoder (v2). Wrist mp4s decoded into zarrs but not consumed
  in v1.
- Multi-suite / multi-task action decoder beyond what fs26 provides. Both
  tasks in fs26 are in scope; nothing else.
- Hyperparameter sweep. Single point (lr=1e-4, layer=20, bsz=128).
- Sim eval. No sim for bi_yams. Real robot only.
- Yams-data video backbone re-finetune. We use ETHRC's iter_7000 as-is.
- `mimic_adapter.py` itself (user owns; I deliver the checkpoint + notes).

---

## What already exists (reused, no new code)

| asset | path | status |
|---|---|---|
| process_bi_yams.py | `data_preprocessing/action/process_bi_yams.py` | EXISTS |
| precompute_t5.py | `data_preprocessing/action/precompute_t5.py` | EXISTS |
| fuse_lora_ckpt.py (with --alpha CLI) | `model/scripts/fuse_lora_ckpt.py` | EXISTS (patched 2026-05-09) |
| bi_yams decoder pipe entry | `world2action_pipe.py:40` (max_horizon=31, in_channels=14) | EXISTS |
| bi_yams dataset / policy_io / top-level configs | `configs/dataloading/{bi_yams.yaml,dataset/bi_yams.yaml,policy_io/bi_yams.yaml}` | EXIST |
| Auto-experiment generator | `configs/experiment/world2action.py` | EXISTS |
| Trainer | `model/scripts/train.py`, `imaginaire/trainer.py` | EXISTS |

**New code total: 1 line** (`world2action_model.py:VIDEO_MODEL_CKPT_NAMES`
gets one entry added).

Plus 1 path edit in `bi_yams.yaml:data_dir`, the fused backbone file in
`checkpoints/video_backbone/`, and the 244 zarrs.

---

## Asks of the user (TBD)

1. AWS p4de.24xlarge provisioning — who's spinning it up, when, and what
   EBS volume size? (Need ≥500 GB for safety.)
2. After training, who rsyncs the iter_NNNN.pt (plus dataset_statistics +
   backbone + tokenizer) from AWS to the robot inference machine?
3. Real-robot eval scheduling — once a checkpoint exists, when's robot time?

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | (codex unavailable in this session) |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR | 4 issues found (3 architecture + 1 code-quality probe); all resolved into the plan. 4 test gaps; all closed. 0 critical failure modes uncovered. |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | n/a (training run, no UI) |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | n/a (research run) |

- **UNRESOLVED:** 0
- **VERDICT:** ENG CLEARED — ready to implement.

### Completion Summary

- Step 0 — Scope Challenge: **scope accepted as-is** (2-file change: 1 line registry + 1 path edit, well under complexity threshold)
- Architecture Review: **4 issues found**
  - Issue 1: throughput is projected not measured → added 50-iter 8-GPU probe (Step 1.5)
  - Issue 2: data prep venue → moved from 5090-then-rsync to AWS-direct (Step 2 rewritten)
  - Issue 3: multi-task embedding handling → self-resolved (precompute_t5.py reads per-zarr prompts when --prompt is omitted)
  - Issue 4: inference deliverable needs full bundle (backbone + decoder + dataset_statistics + tokenizer), not just decoder → Step 6 deliverable section expanded
- Code Quality Review: **0 issues** (plan adds 1 line of Python)
- Test Review: 16 phases diagrammed, **4 gaps identified, all closed**
  - GAP-1 closed: multi-task embedding distinctness check added to Step 2.4
  - GAP-2 closed: 50-iter 8-GPU throughput probe added as Step 1.5
  - GAP-3 closed: dataset_statistics file check added to Step 1.4 smoke acceptance
  - GAP-4 closed: ≥500 GB EBS noted in disk budget section
- Performance Review: 1 medium-confidence note (dataset_statistics auto-gen); self-resolved by existing smoke gate
- NOT in scope: written
- What already exists: written
- TODOS.md updates: 0 (all action items in the plan, none deferred outside it)
- Failure modes: 5 risks listed; all have written mitigations; 0 silent failures
- Outside voice: skipped by user
- Parallelization: **sequential implementation, no parallelization opportunity** (single workstream; data prep + train + eval are inherently sequential)
- Lake Score: high — all complete options chosen at each decision point


