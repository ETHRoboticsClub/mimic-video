import argparse
import pathlib
import pickle

import numpy as np
import tqdm
import zarr
from numcodecs import Blosc

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig
from imaginaire.constants import T5_MODEL_DIR


def log(message: str) -> None:
    print(f"[precompute_t5] {message}", flush=True)


def load_instruction(path: pathlib.Path, source: str) -> str:
    root: zarr.Group
    with zarr.open(str(path), "r") as root:
        if source == "attrs":
            instruction = root.attrs.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"{path}: .zattrs must contain a non-empty 'instruction' string")
            return instruction.strip()

        return root["language_instruction"][0].decode("utf-8")


def encode_prompt(encoder: CosmosT5TextEncoder, prompt: str) -> np.ndarray:
    return encoder.encode_prompts(prompt, max_length=512, return_mask=False).cpu().numpy().astype(np.float16)


def load_embedding_cache(cache_path: pathlib.Path | None) -> dict[str, np.ndarray]:
    if cache_path is None or not cache_path.exists():
        if cache_path is not None:
            log(f"no embedding cache found at {cache_path}; starting a new cache")
        return {}
    log(f"loading embedding cache from {cache_path}")
    with cache_path.open("rb") as stream:
        cache = pickle.load(stream)
    if not isinstance(cache, dict):
        raise ValueError(f"{cache_path}: expected a dict cache keyed by instruction")
    log(f"loaded {len(cache)} cached instruction embeddings")
    return cache


def save_embedding_cache(cache_path: pathlib.Path | None, cache: dict[str, np.ndarray]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as stream:
        pickle.dump(cache, stream, protocol=pickle.HIGHEST_PROTOCOL)


def add_t5(path: pathlib.Path, embedding: np.ndarray):
    root: zarr.Group
    with zarr.open(str(path), "r+") as root:
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


def precompute_dataset(
    dataset: pathlib.Path,
    encoder: CosmosT5TextEncoder,
    *,
    instruction_source: str,
    cache: dict[str, np.ndarray],
    cache_path: pathlib.Path | None,
    prompt_override: str | None,
) -> None:
    paths = sorted(pathlib.Path(dataset).glob("**/*.zarr"))
    log(f"found {len(paths)} zarr episodes under {dataset}")

    if prompt_override:
        log("encoding prompt override once")
        prompt_embedding = encode_prompt(encoder, prompt_override)
        for path in tqdm.tqdm(paths, desc="Writing language embeddings."):
            add_t5(path, prompt_embedding)
        return

    log(f"reading instructions from {instruction_source}")
    instructions_by_path = {path: load_instruction(path, instruction_source) for path in paths}
    unique_instructions = sorted(set(instructions_by_path.values()))
    missing_instructions = [instruction for instruction in unique_instructions if instruction not in cache]
    log(
        f"found {len(unique_instructions)} unique instructions "
        f"({len(missing_instructions)} missing from cache)"
    )

    for instruction in tqdm.tqdm(missing_instructions, desc="Encoding unique instructions."):
        cache[instruction] = encode_prompt(encoder, instruction)
        save_embedding_cache(cache_path, cache)

    save_embedding_cache(cache_path, cache)

    for path in tqdm.tqdm(paths, desc="Writing language embeddings."):
        add_t5(path, cache[instructions_by_path[path]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", nargs="+", type=pathlib.Path, required=True)
    p.add_argument("--prompt", required=False)
    p.add_argument(
        "--instruction-source",
        choices=("dataset", "attrs"),
        default="dataset",
        help="Read prompts from language_instruction or from each zarr .zattrs instruction key.",
    )
    p.add_argument("--cache-path", type=pathlib.Path, help="Optional pickle cache keyed by instruction text.")
    args = p.parse_args()

    log(f"using T5 checkpoint directory: {T5_MODEL_DIR}")
    log("loading T5 text encoder; this can take a while")
    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=T5_MODEL_DIR)
    encoder = CosmosT5TextEncoder(config=encoder_config)
    log("loaded T5 text encoder")

    cache = load_embedding_cache(args.cache_path)

    for dataset in args.dataset_path:
        precompute_dataset(
            dataset,
            encoder,
            instruction_source=args.instruction_source,
            cache=cache,
            cache_path=args.cache_path,
            prompt_override=args.prompt,
        )


if __name__ == "__main__":
    main()
