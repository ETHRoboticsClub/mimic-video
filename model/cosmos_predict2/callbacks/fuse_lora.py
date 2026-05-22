from __future__ import annotations

from imaginaire.utils.callback import Callback
from scripts.fuse_lora_ckpt import fuse_ckpt


class FuseLoraCallback(Callback):
    def on_save_checkpoint_success(
        self, iteration: int = 0, elapsed_time: float = 0, checkpoint_path: str | None = None
    ) -> None:
        fuse_ckpt(checkpoint_path)
