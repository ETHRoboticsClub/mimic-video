# bi_yams training — AWS setup notes

Captures what you need ON THE AWS BOX so `yams_train_launch.sh` will run.

## Instance state (already done on 54.89.85.197)

- 8× A100 80 GB, 8× 875 GB NVMe striped to **`/workspace`** (RAID-0, 6.9 TB, 2.2 GB/s)
- AWS DLAMI Amazon Linux 2023, NVIDIA 580 + CUDA 13
- ~/.aws/{config,login} synced from 5090; `region = us-east-1` (yams bucket lives in us-east-1)
- ~/.netrc synced from 5090 (WandB API key)
- Repo cloned: `/workspace/mimic-video/`

## Repo patches applied (Nico: skim these — they're load-bearing)

These are local edits NOT in main. Without them training will not start or will crash.

| File | Change | Why |
|---|---|---|
| `model/scripts/fuse_lora_ckpt.py` | accepts `--alpha N` CLI arg | ETHRC YAMS finetune used `lora_alpha=16` (not the script's hardcoded 32) |
| `model/cosmos_predict2/configs/defaults/world2action_model.py:67` | added `v2w_yams_unified_iter_000007000_fused` to `VIDEO_MODEL_CKPT_NAMES` | registers the YAMS backbone for the experiment-name generator |
| `model/cosmos_predict2/configs/dataloading/dataset/bi_yams.yaml` | `val_episode_indices` trimmed to episodes that exist on disk | dataset has 244 episodes in metadata; ~78 are unconvertible (truncated video / corrupt h264). val list points only at successful ones |
| `model/cosmos_predict2/data/action/chunk_reader.py:~247` | **OOB-safe slicing** patch (clamp + pad with last frame) | the HF `fs26` dataset has heterogeneous zarr lengths; original chunk_reader raised `IndexError` for anchors near the tail. **Without this patch, training crashes inside DataLoader.** |
| `data_preprocessing/action/process_bi_yams.py` | multi-task support + video-length truncation + safe-wrap | fs26 has 2 tasks (carton-box + towel) and some episodes whose video segments extend past EOF |

## Data state on AWS (already in place)

- **166 zarrs** at `/workspace/mimic-video/data/yams_fs26/` (244 in HF metadata — see notes)
  - 24 episodes (170-194) are unrecoverable: their video segments reference frames past EOF of the actual mp4
  - 2 episodes have corrupt h264 packets that decord can't decode
  - 2 episodes lost to partial writes during early debugging
- All 166 have `language_embedding` from precompute_t5 (2 distinct task embeddings verified)
- Fused YAMS backbone: `model/checkpoints/video_backbone/v2w_yams_unified_iter_000007000_fused.pt` (3.7 GB, alpha=16, net.* only)
- Wan tokenizer: `model/checkpoints/video_backbone/tokenizer/tokenizer.pth`

## Source of the backbone

```
s3://ethrc-ml-data-916780037007/robot-learning/cosmos-predict2-yam/outputs/posttraining/video2world_lora/2b_yams/checkpoints/model/iter_000007000.pt
```

Fused with:
```
python model/scripts/fuse_lora_ckpt.py iter_000007000.pt --alpha 16
# then filter to net.* keys only (drops the net_ema.* copies that match upstream fused format)
```

## Training command (the launch script captures this exactly)

```
torchrun --nproc_per_node=8 --master_port=12360 \
  -m scripts.train --config=cosmos_predict2/configs/config.py \
  -- experiment=w2a_bi_yams_v2w_yams_unified_iter_000007000_fused_lr1.000e-04_layer20_bsz128 \
  trainer.max_iter=10000 \
  trainer.run_validation=False \
  trainer.logging_iter=100 \
  trainer.grad_accum_iter=2 \
  checkpoint.save_iter=200 \
  dataloader_train.batch_size=8 \
  dataloader_train.num_workers=4 \
  dataloader_train.prefetch_factor=2 \
  optimizer.lr=1.0e-04
```

## Why micro=8 grad_accum=2 (not micro=16)

micro=16 on 80 GB cards hits OOM at first iter (peak ~80.1 GB, card has 79.25 GB usable).
micro=8 + grad_accum=2 keeps the same effective bsz=128 with ~half the activation memory.

## Measured throughput

- 27 s/iter steady (8 GPUs, micro=8, grad_accum=2, bsz=128)
- 10K iters ≈ **75 hours ≈ 3.1 days**
- First checkpoint at iter 200 ≈ **90 min from launch**
- 50 checkpoints over the run

## Resume / continuation

The trainer auto-resumes from the latest checkpoint if you re-run with the same EXP name. To continue past 10K, just relaunch with a higher `trainer.max_iter`.

## Inference bundle (deliverable to robot box)

When grabbing a checkpoint for `mimic-yams/mimic_adapter.py`, you need ALL of these:

1. `checkpoints/video_backbone/v2w_yams_unified_iter_000007000_fused.pt` (4 GB)
2. `checkpoints/vam/bi_yams/<EXP>/checkpoints/model/iter_NNNN.pt` (~1 GB)
3. `checkpoints/dataset_statistics/bi_yams.json` (auto-generated during training; CRITICAL — needed to de-normalize joint actions)
4. `checkpoints/video_backbone/tokenizer/tokenizer.pth`

Reference loader code: `eval/libero/run.py:140-191` (`VAMInference.step`). bi_yams is joint-space (14-dim) instead of EE-space; adapt accordingly.
