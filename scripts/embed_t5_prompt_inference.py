"""Encode a single prompt with T5 in the inference-time layout used by
`obs/language_embedding`: a pickled numpy.float32 array of shape (1, 512, 1024).

Usage:
    python -m scripts.embed_t5_prompt_inference \
        --config configs/so101/idm_large_all_tasks.yaml \
        --prompt "the robot arm gently nudges the white object towards the target circle" \
        [--output my_prompt.pickle]
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np

from scripts.so101_pipeline import encode_instruction, load_yaml


def slugify(text: str, max_len: int = 120) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug[:max_len] or "prompt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    text_encoder_cfg = config["preprocessing"]["text_encoder"]
    _trimmed, padded = encode_instruction(args.prompt, text_encoder_config=text_encoder_cfg)
    arr = padded.astype(np.float32)
    assert arr.shape == (1, 512, 1024), f"unexpected shape {arr.shape}"

    output = args.output or Path(f"{slugify(args.prompt)}.pickle")
    with output.open("wb") as f:
        pickle.dump(arr, f)
    print(f"Wrote {output} (shape={arr.shape}, dtype={arr.dtype}).", flush=True)


if __name__ == "__main__":
    main()
