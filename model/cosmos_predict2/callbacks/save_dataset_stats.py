"""Persist per-stream dataset normalization stats to the run's checkpoint
directory at train start. Inference / post-hoc analysis can load this to
denormalize action predictions without re-running statistics over the dataset."""

from __future__ import annotations

import json
import os

import numpy as np

from imaginaire.utils import log
from imaginaire.utils.callback import Callback
from imaginaire.utils.distributed import rank0_only


class SaveDatasetStats(Callback):
    def __init__(self, *args, filename: str = "dataset_stats.json", **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename

    @rank0_only
    def on_train_start(self, model, iteration: int = 0) -> None:
        stats = getattr(model, "dataset_stats", None)
        if stats is None:
            inner = getattr(model, "module", None)
            stats = getattr(inner, "dataset_stats", None) if inner is not None else None
        if not stats:
            return

        out_dir = os.path.join(self.config.job.path_local, "checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, self.filename)

        serializable = {}
        for key, entry in stats.items():
            if not isinstance(entry, dict):
                continue
            serializable[key] = {
                k: np.asarray(v).tolist() if v is not None else None
                for k, v in entry.items()
            }
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        log.info(f"SaveDatasetStats: wrote {len(serializable)} streams to {out_path}")
