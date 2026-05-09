import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
import tqdm
import zarr
from numcodecs import Blosc

# Apex's FusedRMSNorm CUDA extension is not compiled for sm_120 (RTX 5090).
# transformers' T5 prefers it when apex is installed; swap in a torch-native
# RMSNorm before any T5 model class is instantiated.
from transformers.models.t5 import modeling_t5

if modeling_t5.T5LayerNorm.__module__.startswith("apex"):
    class _RMSNormPure(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            in_dtype = hidden_states.dtype
            x32 = hidden_states.to(torch.float32)
            variance = x32.pow(2).mean(-1, keepdim=True)
            x32 = x32 * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * x32.to(in_dtype)

    modeling_t5.T5LayerNorm = _RMSNormPure

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig
from imaginaire.constants import T5_MODEL_DIR


def add_t5(path: pathlib.Path, encoder: CosmosT5TextEncoder, embedding: np.ndarray | None):
    root: zarr.Group
    with zarr.open(str(path), "r+") as root:
        if embedding is None:
            prompt = root["language_instruction"][0].decode("utf-8")
            embedding = (
                encoder.encode_prompts(prompt, max_length=512, return_mask=False).cpu().numpy().astype(np.float16)
            )

        root.create_dataset(
            "language_embedding",
            shape=(1, 512, 1024),
            dtype="float16",
            chunks=(1, 512, 1024),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
            overwrite=True,
        )
        root.create_dataset(
            "language_embedding_timestamps",
            shape=(1,),
            dtype="uint64",
            chunks=(1,),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
            overwrite=True,
        )

        root["language_embedding"][:] = embedding
        root["language_embedding_timestamps"][:] = np.array([0], dtype=np.uint64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", nargs="+", type=pathlib.Path, required=True)
    p.add_argument("--prompt", required=False)
    args = p.parse_args()

    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=T5_MODEL_DIR)
    encoder = CosmosT5TextEncoder(config=encoder_config)

    embedding = args.prompt and encoder.encode_prompts(args.prompt, max_length=512, return_mask=False).cpu().numpy()

    for dataset in args.dataset_path:
        paths = pathlib.Path(dataset).glob("**/*.zarr")

        for path in tqdm.tqdm(paths, desc="Precomputing language embeddings."):
            add_t5(path, encoder=encoder, embedding=embedding)


if __name__ == "__main__":
    main()
