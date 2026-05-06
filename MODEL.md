# mimic-video modeling

This repository is built on top of [cosmos-predict2](https://github.com/nvidia-cosmos/cosmos-predict2). Most code unrelated to mimic-video was removed and some remaining code was simplified to our setting.

## Training

There are several layers of functionality wrapping the network definitions to handle their training. Inference will use a subset of them.

On the outermost layer, there is a generic training loop / [`'ImaginaireTrainer'`](./model/imaginaire/trainer.py) handling distributed setup, optimizer and scheduler steps, checkpointing, and validation.

The trainer calls `training_step` and `validation_step` of a `ImaginaireModel` (either [`Video2WorldModel`](./model/cosmos_predict2/models/video2world_model.py) or [`World2ActionModel`](./model/cosmos_predict2/models/world2action_model.py)). These classes handle the training objective and compute metrics for logging.

The last layer between the trainer and the network is a `'Pipeline'` that handles everything auxiliary to the network including tokenization, normalization, the flow matching sampling procedure, guardrails, etc. As you will have guessed, there are [`Video2WorldPipeline`](./model/cosmos_predict2/pipelines/video2world.py) and [`World2ActionPipeline`](./model/cosmos_predict2/pipelines/world2action.py) in this repo.

The network itself is just a DiT implementation unaware of the concepts described above, there is again a video DiT (called [`MinimalV1LVGDiT`](./model/cosmos_predict2/models/video2world_dit.py)) and of course our [`World2ActionDIT`](./model/cosmos_predict2/models/world2action_dit.py).

## Inference

Inference code does not instantiate a `...Model`, but works only with an inner core of the stack described above, starting with the `...Pipeline` that handles loading weights and running inference. For video-action inference, there is an additional inference-only [`Video2World2ActionPipeline`](./model/cosmos_predict2/pipelines/video2world2action.py) that composes a [`Video2WorldPipeline`](./model/cosmos_predict2/pipelines/video2world.py) with a [`World2ActionPipeline`](./model/cosmos_predict2/pipelines/world2action.py).

Creating and using such a `World2ActionPipeline` for inference is done in [eval/libero/run.py](./eval/libero/run.py) and [eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py](./eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py).

## Config system

The config system is generally a [hydra](https://hydra.cc) setup but quite convoluted.

Most of the training config uses the python API to register individual config groups and full configs ("experiments") choosing values from each group. [configs/experiment/video2world.py](./model/cosmos_predict2/configs/experiment/video2world.py) registers config combinations to train video models and [configs/experiment/world2action.py](./model/cosmos_predict2/configs/experiment/world2action.py) registers config combinations to train action decoders given a frozen video model.

Video-Action dataloading loads its own hydra config from yaml files living in [configs/dataloading](./model/cosmos_predict2/configs/dataloading) instead. This dataloading config gets resolved as a standalone hydra config and is then inserted into the rest of the training config under the `data_config` group. Each `data_config` chooses a `dataset` specifying how to load and interpret the zarr data living on the disk, and a `policy_io` specifying the target fields that will end up in a training batch.
