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

from cosmos_predict2.utils.torch_future import get_total_norm
from imaginaire.utils import distributed
from imaginaire.utils.callback import Callback


@torch.jit.script
def _fused_nan_to_num(params: list[torch.Tensor]):
    for param in params:
        torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0, out=param)


class GradClip(Callback):
    def __init__(self, clip_norm=1.0, force_finite: bool = True, log_wandb: bool = False):
        self.clip_norm = clip_norm
        self.force_finite = force_finite
        self.log_wandb = log_wandb

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler
        if isinstance(model_ddp, distributed.DistributedDataParallel):
            model = model_ddp.module
        else:
            model = model_ddp
        grads = []
        weights = []
        if self.force_finite:
            param: torch.nn.Parameter
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad)
                    weights.append(param.data)
            _fused_nan_to_num(grads)

        total_grad_norm = model.clip_grad_norm_(self.clip_norm)
        total_weight_norm = get_total_norm(weights, 2.0, False, None)

        if self.log_wandb and distributed.is_rank0():
            wandb.log(
                {
                    "global grad norm": total_grad_norm.item(),
                    "global weight norm": total_weight_norm.item(),
                },
                step=iteration,
            )
