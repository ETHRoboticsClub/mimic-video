1. Activate the project venv
2. Convert a LeRobot dataset to zarr using `process_lerobot.py`.

   The script takes one or more `--dataset REPO_ID TASK_TEXT` pairs.
   Set `TASK_TEXT` to `none` to keep the dataset's own task strings, or provide a custom
   override that will be written as `language_instruction` for every episode.

```bash
data_preprocessing/.venv/bin/python data_preprocessing/action/process_lerobot.py \
  --dataset <REPO_ID> "<task text or none>" \
  --output-dir <path_to_output_zarr_folder> \
  [--overwrite] \
  [--episodes <list_of_indices>]
```

   Example — ETHRC/robot-learning-fs26, keeping the dataset's own task text:
```bash
data_preprocessing/.venv/bin/python data_preprocessing/action/process_lerobot.py \
  --dataset ETHRC/robot-learning-fs26 "none" \
  --output-dir /home/ubuntu/tmp/data/yams_fs26 \
  --overwrite
```

   Example — with a custom task text override:
```bash
data_preprocessing/.venv/bin/python data_preprocessing/action/process_lerobot.py \
  --dataset ETHRC/robot-learning-fs26 "Pick up the towel and fold it" \
  --output-dir /home/ubuntu/tmp/data/yams_fs26 \
  --overwrite
```
3. Precompute language embeddings (run from `model/` with the model venv)
```bash
uv run ../data_preprocessing/action/precompute_t5.py --dataset-path <path_to_lerobot_v3_dataset>
```
For example:
```bash
uv run ../data_preprocessing/action/precompute_t5.py --dataset-path ~/.cache/huggingface/hub/datasets--ETHRC--robot-learning-fs26/snapshots/a49199473f399e962f26777c2ef89d9e19f1a20e
```

4. Run training
```bash
DATA_DIR=<path_to_zarr_root> \
VIDEO_CKPT=<path_to_video_backbone.pt> \
EXPERIMENT=<experiment_name> \
NPROC_PER_NODE=<num_gpus> \
BSZ=<global_batch_size> \
bash train.sh
```

## Training runs

### 2026-05-30 — yams, smoke test
```bash
DATA_DIR=/home/ubuntu/tmp/data/yams_fs26 \
VIDEO_CKPT=/home/ubuntu/mimic-video/model/checkpoints/video_backbone/v2w_yams_unified_iter_000007000_fused.pt \
EXPERIMENT=w2a_yams_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128 \
NPROC_PER_NODE=4 \
SMOKE_TEST=1 \
bash train.sh
```

### 2026-05-30 — yams, BSZ=32, 4×GPU
```bash
DATA_DIR=/opt/dlami/nvme/towel \
VIDEO_CKPT=/home/ubuntu/mimic-video/model/checkpoints/video_backbone/v2w_yams_unified_iter_000007000_fused.pt \
EXPERIMENT=w2a_yams_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128 \
NPROC_PER_NODE=4 \
BSZ=32 \
bash train.sh
```
