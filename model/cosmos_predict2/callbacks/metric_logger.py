# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""WandB metric logging callback for cosmos-predict2 action-decoder training.

Lazily initializes a WandB run on rank 0 (project/run-name/tags from env), then
logs scalar entries of ``output_batch`` to WandB for training (every ``every_n``
steps, under ``train/``) and validation (averaged over the val loop, under
``val/``). No-ops gracefully if wandb is unavailable or init fails.
"""

import os
import time

import torch

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import log
from imaginaire.utils.distributed import rank0_only


def _scalars(output_batch: dict) -> dict[str, float]:
    """Pull plain scalar metrics out of an output_batch (skips tensors with dims, lists, dicts)."""
    scalars: dict[str, float] = {}
    for key, value in output_batch.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            scalars[key] = float(value)
        elif torch.is_tensor(value) and value.ndim == 0:
            scalars[key] = value.item()
    return scalars


class MetricLogger(EveryN):
    """Initializes WandB and logs train/val ``output_batch`` scalars (e.g. action_l1). Rank-0 only."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wandb = None
        self._init_attempted = False
        self._last_step_time = None
        self._val_sums: dict[str, float] = {}
        self._val_count = 0

    @rank0_only
    def _lazy_init(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            import wandb
        except ImportError:
            log.warning("MetricLogger: 'wandb' package not installed; skipping WandB logging.")
            return

        project = os.environ.get("WANDB_PROJECT", "vam")
        run_name = os.environ.get("WANDB_RUN_NAME") or os.environ.get("WANDB_NAME")
        tags_env = os.environ.get("WANDB_TAGS", "")
        tags = [t.strip() for t in tags_env.split(",") if t.strip()] or None

        try:
            wandb.init(project=project, name=run_name, tags=tags, resume="allow", reinit=False)
            self._wandb = wandb
            log.info(f"MetricLogger: wandb.init OK — project={project} run={wandb.run.name} url={wandb.run.url}")
        except Exception as e:
            log.warning(f"MetricLogger: wandb.init failed ({type(e).__name__}: {e}); continuing without WandB.")
            self._wandb = None

    @rank0_only
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        self._lazy_init()
        if self._wandb is None:
            return

        payload = {f"train/{key}": value for key, value in _scalars(output_batch).items()}
        now = time.time()
        if self._last_step_time is not None:
            payload["train/iter_time_s"] = now - self._last_step_time
        self._last_step_time = now

        if payload:
            try:
                self._wandb.log(payload, step=int(iteration))
            except Exception as e:
                log.warning(f"MetricLogger: train log failed ({type(e).__name__}: {e})")

    def on_validation_start(self, model: ImaginaireModel, dataloader_val, iteration: int = 0) -> None:
        self._val_sums = {}
        self._val_count = 0

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        scalars = _scalars(output_batch)
        if not scalars:
            return
        for key, value in scalars.items():
            self._val_sums[key] = self._val_sums.get(key, 0.0) + value
        self._val_count += 1

    @rank0_only
    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if self._val_count == 0:
            return
        self._lazy_init()
        if self._wandb is None:
            return
        means = {f"val/{key}": total / self._val_count for key, total in self._val_sums.items()}
        try:
            self._wandb.log(means, step=int(iteration))
        except Exception as e:
            log.warning(f"MetricLogger: val log failed ({type(e).__name__}: {e})")
