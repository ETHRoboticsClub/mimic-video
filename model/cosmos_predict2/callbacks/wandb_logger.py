# SPDX-License-Identifier: Apache-2.0
"""WandB logging callback for cosmos-predict2 action-decoder training.

Initializes a WandB run on rank 0 at the first training step, and logs loss +
iter time on every step. Reads project/run-name/tags from env (or omitted, in
which case wandb uses its defaults).
"""

import os
import time

import torch
from torch import Tensor

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import log
from imaginaire.utils.distributed import rank0_only


class WandbLogger(EveryN):
    """Logs training loss + iter speed to WandB. Rank-0 only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wandb = None
        self._init_attempted = False
        self._last_step_time = None
        self._name = self.__class__.__name__

    @rank0_only
    def _lazy_init(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            import wandb
        except ImportError:
            log.warning("WandbLogger: 'wandb' package not installed; skipping WandB logging.")
            return

        project = os.environ.get("WANDB_PROJECT", "vam")
        run_name = os.environ.get("WANDB_RUN_NAME") or os.environ.get("WANDB_NAME")
        tags_env = os.environ.get("WANDB_TAGS", "")
        tags = [t.strip() for t in tags_env.split(",") if t.strip()] or None

        try:
            self._wandb = wandb
            wandb.init(
                project=project,
                name=run_name,
                tags=tags,
                resume="allow",
                reinit=False,
            )
            log.info(f"WandbLogger: wandb.init OK — project={project} run={wandb.run.name} url={wandb.run.url}")
        except Exception as e:
            log.warning(f"WandbLogger: wandb.init failed ({type(e).__name__}: {e}); continuing without WandB.")
            self._wandb = None

    @rank0_only
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self._lazy_init()
        if self._wandb is None:
            return

        now = time.time()
        iter_time = None if self._last_step_time is None else (now - self._last_step_time)
        self._last_step_time = now

        # log every iter (cheap, ~ms)
        payload = {
            "train/loss": float(loss.detach().item()),
            "train/iteration": int(iteration),
        }
        if iter_time is not None:
            payload["train/iter_time_s"] = iter_time
        try:
            self._wandb.log(payload, step=int(iteration))
        except Exception as e:
            log.warning(f"WandbLogger: log failed ({type(e).__name__}: {e})")

    def every_n_impl(self, *args, **kwargs) -> None:
        # Not used — we log every step via on_training_step_end. The base class
        # requires this method to exist.
        pass
