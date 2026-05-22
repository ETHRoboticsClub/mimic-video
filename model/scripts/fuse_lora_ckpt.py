import argparse

import torch

DEFAULT_ALPHA = 32


def fuse_ckpt(ckpt_path: str, alpha: float = DEFAULT_ALPHA) -> str:
    # Force CPU load: this is called from a background thread on rank 0 while
    # the main thread is doing validation video generation on GPU. Loading
    # checkpoint tensors back to GPU caused VRAM contention and CUDA stream
    # deadlocks that timed out the NCCL barrier on ranks 1-3.
    ckpt = torch.load(ckpt_path, map_location="cpu")

    lora_rank = None

    for key in list(ckpt.keys()):
        if "lora_A" not in key:
            continue

        b_key = key.replace("lora_A", "lora_B")
        base_key = key.replace("lora_A.default", "base_layer")

        a = ckpt[key].cpu()
        b = ckpt[b_key].cpu()
        base = ckpt[base_key].cpu()

        this_rank = a.shape[0]
        lora_rank = lora_rank or this_rank
        assert lora_rank == this_rank, (lora_rank, this_rank)

        adapter = b @ a
        fused = base + alpha / this_rank * adapter

        del ckpt[key], ckpt[b_key], ckpt[base_key]

        ckpt[base_key.replace(".base_layer", "")] = fused

    print(f"{lora_rank=} {alpha=}")

    fused_path = ckpt_path.replace(".pt", "_fused.pt")
    torch.save(ckpt, fused_path)

    return fused_path


def main():
    parser = argparse.ArgumentParser(description="Fuse adapters into weight.")
    parser.add_argument("ckpt_path", type=str, help="checkpoint path")
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=(
            "LoRA alpha used during training. Must match training-time "
            f"lora_alpha or the fused weights will be numerically wrong "
            f"(default: {DEFAULT_ALPHA})."
        ),
    )
    args = parser.parse_args()

    fuse_ckpt(args.ckpt_path, alpha=args.alpha)


if __name__ == "__main__":
    main()
