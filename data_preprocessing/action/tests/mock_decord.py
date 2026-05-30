"""Minimal decord mock for testing on platforms where decord cannot be built."""

from __future__ import annotations

import pathlib

import imageio.v3 as iio
import numpy as np


class VideoReader:
    """Mock decord.VideoReader backed by imageio."""

    def __init__(self, uri: str | pathlib.Path, ctx=None, num_threads=1):
        self._frames = iio.imread(str(uri))  # (T, H, W, C) uint8
        if self._frames.ndim == 3:
            self._frames = self._frames[np.newaxis]

    def __len__(self) -> int:
        return len(self._frames)

    def get_batch(self, idx: list[int]):
        arr = self._frames[idx]
        return _FakeBatch(arr)


class _FakeBatch:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def asnumpy(self) -> np.ndarray:
        return self._arr


def cpu(device_id: int):
    return None  # ignored


__all__ = ["VideoReader", "cpu"]
