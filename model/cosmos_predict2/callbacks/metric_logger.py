# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import wandb

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import distributed


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
    """Logs scalar entries of ``output_batch`` to wandb.

    Train metrics are logged every ``every_n`` steps under ``train/``. Validation metrics are
    averaged over the validation loop and logged once under ``val/``. No-ops if no wandb run exists.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._val_sums: dict[str, float] = {}
        self._val_count = 0

    @distributed.rank0_only
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if wandb.run is None:
            return
        scalars = {f"train/{key}": value for key, value in _scalars(output_batch).items()}
        if scalars:
            wandb.log(scalars, step=iteration)

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

    @distributed.rank0_only
    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if wandb.run is None or self._val_count == 0:
            return
        means = {f"val/{key}": total / self._val_count for key, total in self._val_sums.items()}
        wandb.log(means, step=iteration)
