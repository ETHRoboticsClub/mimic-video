import os
from collections.abc import Mapping
from numbers import Number
from typing import Any

import torch
import wandb
from omegaconf import DictConfig, ListConfig

from imaginaire.utils import distributed, log
from imaginaire.utils.callback import Callback


def _to_wandb_config(value: Any) -> Any:
    if isinstance(value, DictConfig | ListConfig):
        value = value.copy()
    if isinstance(value, Mapping):
        return {str(k): _to_wandb_config(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_wandb_config(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


def _to_scalar(value: Any) -> float | int | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return value.detach().float().item()
    if isinstance(value, Number):
        return value
    return None


def _flatten_metrics(prefix: str, value: Any, metrics: dict[str, float | int]) -> None:
    scalar = _to_scalar(value)
    if scalar is not None:
        metrics[prefix] = scalar
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten_metrics(f"{prefix}/{key}", item, metrics)
        return

    if isinstance(value, list | tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 for item in value):
            values = [_to_scalar(item[1]) for item in value]
            values = [item for item in values if item is not None]
            if values:
                metrics[f"{prefix}/mean"] = sum(values) / len(values)


class WandbCallback(Callback):
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.validation_step = 0

    @distributed.rank0_only
    def on_train_start(self, model, iteration: int = 0) -> None:
        del model
        if not self.enabled:
            return

        if "WANDB_API_KEY" not in os.environ and "WANDB_KEY" in os.environ:
            os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]

        try:
            wandb.init(
                project=self.config.job.project,
                group=self.config.job.group,
                name=self.config.job.name,
                config=_to_wandb_config(self.config.to_dict()),
                resume="allow",
            )
        except Exception as exc:
            log.warning(f"W&B initialization failed; continuing without W&B logging: {exc}")
            self.enabled = False

    @distributed.rank0_only
    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del model, data_batch
        if not self.enabled or wandb.run is None:
            return
        metrics: dict[str, float | int] = {"train/loss_tensor": loss.detach().float().item()}
        _flatten_metrics("train", output_batch, metrics)
        wandb.log(metrics, step=iteration)

    @distributed.rank0_only
    def on_validation_start(self, model, dataloader_val, iteration: int = 0) -> None:
        del model, dataloader_val, iteration
        self.validation_step = 0

    @distributed.rank0_only
    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        del model, data_batch
        if not self.enabled or wandb.run is None:
            return
        metrics: dict[str, float | int] = {
            "validation/loss_tensor": loss.detach().float().item(),
            "validation/global_step": iteration,
            "validation/batch_idx": self.validation_step,
        }
        _flatten_metrics("validation", output_batch, metrics)
        wandb.log(metrics)
        self.validation_step += 1

    @distributed.rank0_only
    def on_train_end(self, model, iteration: int = 0) -> None:
        del model, iteration
        if wandb.run is not None:
            wandb.finish()
