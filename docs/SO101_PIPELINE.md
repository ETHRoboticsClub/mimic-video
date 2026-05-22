# SO101 YAML Pipeline

This document captures the SO101 Hugging Face preprocessing and smoke training path added for `rl26-world-models/so101-max-speed`, plus the model-training details that are easy to confuse.

## Entry Point

Run the pipeline from the repository root:

```bash
model/.venv/bin/python scripts/so101_pipeline.py --config configs/so101/smoke.yaml
```

The launcher reads one YAML file with two top-level sections:

```yaml
preprocessing:
  source:
    repo_id: rl26-world-models/so101-max-speed
    revision: main
  text_encoder:
    model_id: google-t5/t5-large
    revision: main
  instruction: push the object to the right
  video_feature: observation.images.top
  exclude_episodes: []
  image:
    crop_xywh: [0, 0, 640, 480]
    output_size_hw: [480, 640]
  contact_sheet:
    enabled: true
  force: false

training:
  video_finetune:
    enabled: true
    experiment: so101
    trainer:
      max_iter: 10
    checkpoint:
      save_iter: 0
  action_decoder:
    enabled: true
    experiment: so101
    trainer:
      max_iter: 10
    checkpoint:
      save_iter: 0
```

The YAML is the user-facing config. The launcher injects the resolved data paths through Hydra; smoke runs use the same schema as full runs with smaller values such as `trainer.max_iter: 10` and `checkpoint.save_iter: 0`.

For multi-task video-backbone preprocessing, replace the single `source`/`instruction`/`video_feature` fields with `sources`. Each source is converted into one merged cache with collision-free `sourceNNN_` stems, and each source's instruction is encoded separately:

```yaml
preprocessing:
  sources:
    - repo_id: org/so101-task-a
      revision: main
      instruction: push the object to the goal
      video_feature: observation.images.front
      exclude_episodes: []
    - repo_id: org/so101-task-b
      revision: main
      instruction: move the cube left
      video_feature: observation.images.front
      exclude_episodes: []
  text_encoder:
    model_id: google-t5/t5-large
    revision: main
  image:
    crop_xywh: [0, 0, 1280, 720]
    output_size_hw: [480, 640]
```

For an actual action-decoder/IDM training run, use:

```bash
model/.venv/bin/python scripts/so101_pipeline.py --config configs/so101/idm_train.yaml
```

`idm_train.yaml` uses the same launcher path as `smoke.yaml`, but with production-sized values and `launch.dryrun: false`. It exposes the action-decoder launch, job, dataset split, trainer, checkpoint, train and validation dataloaders, optimizer, learning-rate scheduler, model, action scheduler, EMA, and network hyperparameters directly in YAML. The action-decoder model section includes `validation_mode`: `full` preserves the original GT-video plus generated-video validation, `gt_video` validates only supervised action prediction from ground-truth video activations, and `generated_video` validates from predicted-video activations. The checked-in IDM config enables validation with `validation_mode: gt_video`, `max_val_iter: 30`, and `dataloader_val.batch_size: 8`, so validation uses held-out ground-truth trajectories without generating video. The launcher still handles preprocessing, revision reuse, and T5 embedding generation, then passes those YAML values through as Hydra overrides.

Checkpoint retention is controlled with `checkpoint.keep_latest`. Set it to an integer to keep only that many newest local checkpoints after each successful save, or `null` to keep all checkpoints. `checkpoint.save_at_epoch_end` gates the trainer's extra epoch-boundary checkpoint saves; set it to `false` when short distributed epochs would otherwise save too often. `trainer.validate_at_epoch_end` gates extra validation after each completed dataloader epoch; set it to `false` when validation should follow `trainer.validation_iter` only.

Validation can use random held-out episodes or exact timestamp windows. Leave `training.action_decoder.dataset.val_episode_ranges: []` to use `num_val_episodes`. To validate only known segments, set ranges in seconds:

```yaml
training:
  action_decoder:
    dataset:
      num_val_episodes: 0
      val_episode_ranges:
        - episode: 3
          start_s: 4.0
          end_s: 9.0
        - episode: 17
          start_s: 1.5
          end_s: 6.0
```

When explicit ranges are set, the listed episodes are removed entirely from the training split, and validation only reads chunks whose anchor timestamp is inside the requested windows.

## Authentication

For private Hugging Face datasets, set one of:

```bash
export HF_TOKEN=...
export HUGGINGFACE_HUB_TOKEN=...
```

For W&B, set:

```bash
export WANDB_API_KEY=...
export WANDB_MODE=online
```

The repository ignores `.env`, so credentials can live there locally. The launcher uses the active environment; it does not read `.env` by itself unless the shell sources it first.

## Revisioned Data Cache

Single-source preprocessed data is stored under:

```text
data/<repo_id>/<numeric_revision>/
```

For the current dataset this looks like:

```text
data/rl26-world-models/so101-max-speed/2/
```

The launcher normalizes the `preprocessing` section, excluding `force`, and writes it to:

```text
data/<repo_id>/<revision>/preprocessing.yaml
```

Merged multi-source preprocessed data is stored under:

```text
data/_so101_mixes/<numeric_revision>/
```

That cached YAML includes:

- pipeline name and version
- Hugging Face dataset `repo_id`, requested revision, and resolved commit SHA, or a `sources` list with one entry per merged dataset
- T5 text encoder model id, requested revision, and resolved commit SHA
- instruction text, selected LeRobot video feature, and sorted `exclude_episodes` per source
- fixed crop and output size
- contact sheet setting

Revision selection:

- If no numeric revision exists, create `1`.
- If an existing `preprocessing.yaml` exactly matches the normalized current config, reuse that revision.
- If no match exists, create `max(existing_numeric_revision) + 1`.
- If `force: true` and a matching revision exists, overwrite that revision.
- If Hugging Face dataset commit resolution fails but a local cached revision for that dataset/revision has a commit SHA, the launcher uses that cached commit so local training can continue.

`exclude_episodes` replaced the earlier `include_episodes` idea. The default `[]` converts all episodes.

## Preprocessing Outputs

Each revision is laid out as:

```text
data/<repo_id>/<revision>/
  preprocessing.yaml
  contact_sheet.jpg
  video_finetune/
    video/
      episode_000000.mp4
    metas/
      episode_000000.txt
    t5_xxl/
      episode_000000.pickle
    vae_latents/        # optional, when precompute_vae_latents is enabled
      episode_000000.pt
  action_decoder/
    episode_000000.zarr
```

For merged sources, filenames are prefixed:

```text
data/_so101_mixes/<revision>/
  video_finetune/video/source000_episode_000000.mp4
  video_finetune/t5_xxl/source000_episode_000000.pickle
  action_decoder/source000_episode_000000.zarr
```

The converter reads the LeRobot-style Hugging Face repository using:

- `meta/info.json` for FPS, feature names, feature shapes, and video feature discovery
- `meta/episodes/*.parquet` for episode boundaries and shared video shard offsets
- `data/chunk-000/file-*.parquet` for per-frame `observation.state`, `action`, timestamps, frame indices, and episode indices
- `videos/<preprocessing.video_feature>/chunk-000/file-*.mp4` for frames

The current implementation uses Hugging Face Hub, PyArrow/Pandas, PyAV, PIL, NumPy, and Zarr. It does not currently use LeRobot APIs or NVIDIA video tooling.

SO101 preprocessing requires `meta/info.json` to declare `fps: 10`; higher-rate datasets must be downsampled before this launcher. Frame decoding uses each episode's video shard offset plus frame indices derived from episode-relative row timestamps on the 10fps grid, rather than trusting the potentially ambiguous parquet `frame_index` column. Those derived frame indices must start at frame 0, be finite, non-negative, contiguous, strictly increasing, and aligned to 10fps; zarr timestamps are then quantized from the same indices. Any mismatch raises a `ValueError` naming the episode, row offsets, timestamps, and computed frame indices.

Frames are cropped with fixed `crop_xywh`, resized to `output_size_hw`, and written to both:

- `video_finetune/video/*.mp4`
- `action_decoder/*.zarr` as `workspace_rgb`

The current Cosmos training path requires `output_size_hw: [480, 640]`; the config field exists, but other sizes intentionally raise until the downstream model path is changed.

## Action Decoder Zarr Format

For each episode zarr:

```text
workspace_rgb                  uint8,   shape (T, 480, 640, 3)
workspace_rgb_timestamps       uint64,  shape (T,)
joint_state_lowdim             float32, shape (T, 6)
joint_state_lowdim_timestamps  uint64,  shape (T,)
joint_action_lowdim            float32, shape (T, 6)
joint_action_lowdim_timestamps uint64,  shape (T,)
language_instruction           bytes,   shape (1,)
language_instruction_timestamps uint64, shape (1,)
language_embedding             float16, shape (1, 512, 1024)
language_embedding_timestamps  uint64,  shape (1,)
```

The mapping is:

- `observation.state` -> `joint_state_lowdim`
- `action` -> `joint_action_lowdim`

The current SO101 pipeline expects both arrays to be 6D. Changing this action/state space is a model and data contract change, not just a preprocessing option.

Validation checks:

- matching frame/state/action/timestamp lengths
- strictly increasing `uint64` nanosecond timestamps
- image shape `(480, 640, 3)`
- low-dimensional state/action shape `(T, 6)`
- matching episode counts between video finetune and action decoder outputs

Timing failures are intended to be loud and specific during conversion: 30fps source metadata, off-grid timestamps, duplicate timestamps, non-contiguous timestamp-derived frames, non-numeric `from_timestamp`, and missing decoded source frames all raise direct exceptions instead of producing a silently mis-timed cache.

## T5 Embeddings

The `t5_xxl` directory name is inherited from the upstream Cosmos/mimic-video data format. These files are text prompt embeddings, not video embeddings.

For single-source configs, the launcher encodes the YAML `preprocessing.instruction` once with T5, then writes the same prompt embedding into:

- `video_finetune/t5_xxl/*.pickle` for video finetuning
- `action_decoder/*.zarr/language_embedding` for action decoder training

For multi-source configs, it encodes each `preprocessing.sources[*].instruction` separately and writes that embedding only to episodes whose filename has the matching `sourceNNN_` prefix.

For the L40S smoke config, `google-t5/t5-large` is used because it emits the expected 1024-dimensional embeddings while avoiding the local T5-11B memory spike. Switching back to T5-11B should use a sharded or safetensors checkpoint and a host with enough RAM.

## Training Integration

The launcher runs each enabled stage as:

1. Hydra dry run to verify config composition.
2. Real `torchrun --nproc_per_node=1` training run.

Smoke behavior is configured in YAML instead of through a separate launcher mode. For example, the smoke config sets:

```text
trainer.max_iter=10
trainer.logging_iter=1
trainer.run_validation=False
trainer.validation_iter=999999999
checkpoint.save_iter=0
dataloader_train.batch_size=1
dataloader_train.num_workers=0
dataloader_train.persistent_workers=False
dataloader_train.prefetch_factor=null
trainer.callbacks.wandb.enabled=True
```

The single-process dataloader overrides were added after a 10-step action-decoder run was SIGKILLed with the default 12 workers and prefetching. With `num_workers=0`, the 10-step action smoke run completed on the L40S.

The launcher does not apply hidden smoke overrides. Checkpoint frequency, validation, dataloader settings, optimizer settings, scheduler settings, and action-decoder network size come from the YAML.

When `training.video_finetune.dataset.precompute_vae_latents` is enabled, the launcher stores Cosmos VAE latents in `video_finetune/vae_latents`. The video dataset then emits `video_latent` batches instead of raw `video` tensors, and Video2World training treats those cached latents as video batches while still rejecting mixed image/video/latent inputs.

Video-backbone runs can generate one deterministic validation sample at the end of each validation pass:

```yaml
training:
  video_finetune:
    eval_video:
      enabled: true
      episode_index: 17        # single-source configs
      # episode_stem: source000_episode_000017  # merged-source configs
      seed_start_frame: 0
      num_seed_frames: 5
      seed: 0
      guidance: 7.0
      num_sampling_step: 35
```

The selected episode or merged-source `episode_stem` must belong to the same deterministic filename-hash validation split as `training.video_finetune.dataset.val_ratio`; otherwise training fails with a clear error. Generated videos are saved under the run directory's `video_eval/` folder and logged to W&B when W&B is enabled.

## W&B

W&B support is implemented as a generic rank-0 training callback:

- registered in `model/cosmos_predict2/configs/defaults/callbacks.py`
- implemented in `model/imaginaire/utils/callback.py`
- disabled by default
- enabled with `trainer.callbacks.wandb.enabled=True`

The callback initializes:

```text
project = job.project
group   = job.group
name    = job.name
dir     = job.path_local
```

It logs scalar training loss as:

```text
train/loss
iteration
```

The SO101 YAML defaults route:

- video backbone smoke runs to `rl-mimic-backbone`
- action decoder smoke runs to `rl-mimic-idm`

Verified W&B syncs from May 11, 2026:

- video backbone: `rl-mimic-backbone`, run `uyam5q06`
- action decoder: `rl-mimic-idm`, run `wg37kl8g`

## Video Backbone Training

For SO101, `training.video_finetune.experiment: so101` resolves to:

```text
v2w_so101_lora_rank256_lr1.778e-04_bsz32
```

That is LoRA finetuning of the pretrained Cosmos video backbone, not full finetuning. The video model path:

- loads `v2w_pretrained_cosmos.pt`
- freezes the existing video pipeline first
- injects LoRA adapters into the video DiT
- unfreezes `adaln_modulation` parameters as part of the LoRA path

The rank is currently 256 and the LoRA target modules are configured in `model/cosmos_predict2/configs/experiment/video2world.py`.

## Action Decoder Training

For SO101, `training.action_decoder.experiment: so101` resolves to:

```text
w2a_so101_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1
```

The action decoder is trained with a frozen video backbone. It loads the pretrained video DiT, freezes it, extracts/uses video latent features, and trains the `World2ActionDIT` action denoising model.

The SO101 action decoder is intentionally small for smoke testing:

```text
in_channels=6
out_channels=6
model_channels=256
num_blocks=4
num_heads=4
max_horizon=61
```

That is why the smoke run reported about 9.2M trainable parameters. The total model log also includes the frozen 2B video backbone, so it reported roughly:

```text
Total parameters: 1.97B
Frozen parameters: 1,956,413,440
Trainable parameters: 9,223,858
```

The larger Bridge/LIBERO action-decoder configs use 1024 channels and 24 transformer blocks. The SO101 smoke config does not use that larger decoder.

## Is The Action Decoder Pretrained?

In the paper/repo design, the action decoder is trained from scratch unless an action decoder checkpoint is explicitly loaded.

The pretrained part is the video backbone. The action decoder is a DiT-style transformer over low-dimensional action/state tokens, conditioned on frozen video model features. If no `action_dit_path` is provided, the `World2ActionPipeline` initializes the action DiT with random weights and logs that it is initializing from random weights.

The released Bridge and LIBERO action decoders are pretrained only in the ordinary sense that the authors trained and released those checkpoints. They do not come from an external pretrained vision transformer.

## Implemented Files

User-facing entry points:

- `configs/so101/smoke.yaml`
- `configs/so101/idm_train.yaml`
- `scripts/so101_pipeline.py`
- `docs/SO101_PIPELINE.md`

SO101 data/model configs:

- `model/cosmos_predict2/configs/dataloading/so101.yaml`
- `model/cosmos_predict2/configs/dataloading/dataset/so101.yaml`
- `model/cosmos_predict2/configs/dataloading/policy_io/so101.yaml`
- `model/cosmos_predict2/configs/dataloading/dataset/transform/so101_to_so101.yaml`
- `model/cosmos_predict2/configs/defaults/data_video.py`
- `model/cosmos_predict2/configs/defaults/data_action.py`
- `model/cosmos_predict2/configs/defaults/world2action_pipe.py`
- `model/cosmos_predict2/configs/defaults/video2world_model.py`
- `model/cosmos_predict2/configs/experiment/video2world.py`

Training/W&B integration:

- `model/imaginaire/utils/callback.py`
- `model/cosmos_predict2/configs/defaults/callbacks.py`
- `model/imaginaire/trainer.py`

Documentation and local hygiene:

- `README.md`
- `INFO.md`
- `.gitignore`
- `.env.example`

## Not Implemented Yet

These were discussed but are not part of the current implementation:

- random crop augmentation
- explicit area-of-interest schema beyond fixed `crop_xywh`
- automatic workspace detection
- output resolutions other than `[480, 640]`
- LeRobot API based decoding
- NVIDIA GPU video decode/encode path
- SO101 full-size action decoder training
- SO101 action-decoder LoRA
