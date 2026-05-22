# mimic-video modeling

This repository is built on top of [cosmos-predict2](https://github.com/nvidia-cosmos/cosmos-predict2). Most code unrelated to mimic-video was removed and some remaining code was simplified to our setting.

## Training

There are several layers of functionality wrapping the network definitions to handle their training. Inference will use a subset of them.

On the outermost layer, there is a generic training loop / [`'ImaginaireTrainer'`](./model/imaginaire/trainer.py) handling distributed setup, optimizer and scheduler steps, checkpointing, and validation. Validation cadence is controlled by `trainer.validation_iter`; the old extra end-of-epoch validation path is intentionally commented out to avoid surprise validation runs and memory spikes when an epoch boundary lands near a scheduled validation step.

Distributed setup loads CUDA runtime through the system linker when available, and falls back to the `nvidia-cuda-runtime-cu12` wheel's `libcudart.so.12` so `uv sync --extra cu126` environments do not require a system CUDA toolkit symlink.

Training callbacks are registered through the Cosmos defaults. The optional W&B callback is disabled by default and can be enabled with Hydra override `trainer.callbacks.wandb.enabled=True`; it initializes a rank-0 run from `job.project`, `job.group`, and `job.name`, sends the full serialized training config to W&B, then logs scalar training and validation loss.

Checkpoint saving supports optional local retention through `checkpoint.keep_latest`. When set, the checkpointer prunes older `iter_*.pt` checkpoints after a new checkpoint has been saved; `None`/`null` keeps all checkpoints. `checkpoint.save_at_epoch_end` controls the trainer's additional epoch-boundary checkpoint saves near 10k iteration boundaries. `trainer.validate_at_epoch_end` controls whether validation also runs after each completed training dataloader epoch in addition to `trainer.validation_iter`.

The trainer calls `training_step` and `validation_step` of a `ImaginaireModel` (either [`Video2WorldModel`](./model/cosmos_predict2/models/video2world_model.py) or [`World2ActionModel`](./model/cosmos_predict2/models/world2action_model.py)). These classes handle the training objective and compute metrics for logging.

The last layer between the trainer and the network is a `'Pipeline'` that handles everything auxiliary to the network including tokenization, normalization, the flow matching sampling procedure, guardrails, etc. As you will have guessed, there are [`Video2WorldPipeline`](./model/cosmos_predict2/pipelines/video2world.py) and [`World2ActionPipeline`](./model/cosmos_predict2/pipelines/world2action.py) in this repo.

The network itself is just a DiT implementation unaware of the concepts described above, there is again a video DiT (called [`MinimalV1LVGDiT`](./model/cosmos_predict2/models/video2world_dit.py)) and of course our [`World2ActionDIT`](./model/cosmos_predict2/models/world2action_dit.py).

## Inference

Inference code does not instantiate a `...Model`, but works only with an inner core of the stack described above, starting with the `...Pipeline` that handles loading weights and running inference. For video-action inference, there is an additional inference-only [`Video2World2ActionPipeline`](./model/cosmos_predict2/pipelines/video2world2action.py) that composes a [`Video2WorldPipeline`](./model/cosmos_predict2/pipelines/video2world.py) with a [`World2ActionPipeline`](./model/cosmos_predict2/pipelines/world2action.py).

Creating and using such a `World2ActionPipeline` for inference is done in [eval/libero/run.py](./eval/libero/run.py) and [eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py](./eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py).

## Action decoder outputs

The inverse-dynamics/action-decoder side of the model is implemented by `World2ActionPipeline`. At inference time it returns an unnormalized `action/lowdim_concat` tensor with shape `(B, H_A, A)`, where `B` is batch size, `H_A` is the predicted action horizon, and `A` is the action dimension. For the released Bridge and LIBERO policies, `A = 10`.

Action-decoder validation is controlled by `model.config.validation_mode`. The default `full` mode preserves the original validation behavior, computing GT-video/noise metrics and generated-video metrics. `gt_video` validates only the supervised action-decoder objective from held-out ground-truth video activations, while `generated_video` keeps the predicted-video validation branch without the GT-video metric sweep. SO101 action-decoder runs can either use random whole-episode validation via `training.action_decoder.dataset.num_val_episodes` or explicit timestamp windows via `training.action_decoder.dataset.val_episode_ranges`; when ranges are set, training excludes the listed episodes entirely and validation reads only chunks whose anchor timestamp falls inside those windows.

Each predicted action timestep is laid out as:

```
action[0:3]  end-effector translation command: x, y, z
action[3:9]  end-effector rotation command as a 6D rotation representation
action[9]    gripper command
```

The rotation command is still a normal 3D end-effector orientation; it is just represented with six numbers. Those six numbers are the first two rows of a 3x3 rotation matrix:

```
action[3:6]  first row of the rotation matrix
action[6:9]  second row of the rotation matrix
```

At execution time the first row is normalized, the second row is made orthogonal to the first and normalized, and the third row is reconstructed with a cross product. This gives a valid 3x3 rotation matrix. The model uses this 6D representation because it is easier for a neural network to regress than Euler angles, quaternions, or axis-angle: it avoids angle wrapping, gimbal lock, and quaternion sign ambiguity while using fewer values than a full 9D rotation matrix.

The exact action semantics differ slightly by benchmark. For Bridge, the first nine values are interpreted as a relative end-effector target pose, and the gripper value is trained as a clipped action in roughly `[0, 1]` before evaluation maps it to the simulator command range. For LIBERO, the first three values are an xyz delta, the rotation is a relative rotation delta encoded in 6D form, and the gripper command is interpreted by sign: negative opens and positive closes.

## Config system

The config system is generally a [hydra](https://hydra.cc) setup but quite convoluted.

Most of the training config uses the python API to register individual config groups and full configs ("experiments") choosing values from each group. [configs/experiment/video2world.py](./model/cosmos_predict2/configs/experiment/video2world.py) registers config combinations to train video models and [configs/experiment/world2action.py](./model/cosmos_predict2/configs/experiment/world2action.py) registers config combinations to train action decoders given a frozen video model.

Video-Action dataloading loads its own hydra config from yaml files living in [configs/dataloading](./model/cosmos_predict2/configs/dataloading) instead. This dataloading config gets resolved as a standalone hydra config and is then inserted into the rest of the training config under the `data_config` group. Each `data_config` chooses a `dataset` specifying how to load and interpret the zarr data living on the disk, and a `policy_io` specifying the target fields that will end up in a training batch.

The SO101 Hugging Face preprocessing and training path is documented in [docs/SO101_PIPELINE.md](./docs/SO101_PIPELINE.md). It covers the YAML launcher, revisioned local data cache, explicit LeRobot `preprocessing.video_feature` camera selection, fixed-crop conversion format, W&B runs, and the distinction between LoRA video-backbone finetuning and from-scratch action-decoder training. The launcher now enforces `meta/info.json` `fps: 10`, derives decoded frame indices from episode-relative row timestamps instead of ambiguous parquet `frame_index` values, requires those timestamp-derived frames to start at 0 and be contiguous, and writes zarr timestamps quantized onto the same 10fps grid with direct `ValueError` messages for off-grid, duplicate, non-monotonic, or missing-frame timing. The launcher can preprocess one source under `data/<repo_id>/<revision>/` or merge several `preprocessing.sources` into one ordinary dataloader directory under `data/_so101_mixes/<revision>/`, with `sourceNNN_` filename prefixes and per-source T5 instruction embeddings; [configs/so101/video_finetune.yaml](./configs/so101/video_finetune.yaml) is configured as a two-source trimmed whole-arm video-finetune mix. The launcher prints progress through HF commit resolution, cache hits/misses, episode conversion, cache validation, T5 embedding generation, and the training handoff so stalled runs show which stage they are in. The smoke path uses [configs/so101/smoke.yaml](./configs/so101/smoke.yaml); the real IDM/action-decoder path uses [configs/so101/idm_train.yaml](./configs/so101/idm_train.yaml), which exposes the trainer, checkpoint, dataloader, optimizer, scheduler, model, action scheduler, EMA, and network hyperparameters as YAML options.

SO101 video-backbone finetuning can precompute Cosmos VAE latents under `video_finetune/vae_latents`. When present, the video dataset emits `video_latent` batches instead of raw `video` tensors, and the Video2World model/pipeline treat those cached latents as the video input path while still rejecting mixed image/video/latent batches.

SO101 video-backbone validation can also generate a deterministic sample video through `training.video_finetune.eval_video`. The callback runs at validation end, verifies that the configured `episode_index` or merged-dataset `episode_stem` is in the same filename-hash validation split as the dataloader, uses the configured seed-frame window and sampling seed, writes the result under the job's `video_eval/` directory, and logs it to W&B when available.
