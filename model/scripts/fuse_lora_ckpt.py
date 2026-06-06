import argparse

import torch

ALPHA = 32


def fuse_ckpt(ckpt_path: str, alpha: float = ALPHA) -> str:
    ckpt = torch.load(ckpt_path)

    lora_rank = None

    for key in list(ckpt.keys()):
        if "lora_A" not in key:
            continue

        b_key = key.replace("lora_A", "lora_B")
        base_key = key.replace("lora_A.default", "base_layer")

        this_rank = ckpt[key].shape[0]
        lora_rank = lora_rank or this_rank
        assert lora_rank == this_rank, (lora_rank, this_rank)
        assert alpha == this_rank, (
            f"alpha ({alpha}) != rank ({this_rank}); fuse scale would be {alpha / this_rank} "
            f"instead of 1.0. Pass --alpha matching the trained lora_alpha."
        )

        adapter = ckpt[b_key] @ ckpt[key]
        fused = ckpt[base_key] + alpha / this_rank * adapter

        del ckpt[key], ckpt[b_key], ckpt[base_key]

        ckpt[base_key.replace(".base_layer", "")] = fused

    print(f"{lora_rank=}")

    fused_path = ckpt_path.replace(".pt", "_fused.pt")
    torch.save(ckpt, fused_path)

    return fused_path


def main():
    parser = argparse.ArgumentParser(description="Fuse adapters into weight.")
    parser.add_argument("ckpt_path", type=str, help="checkpoint path")
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="LoRA alpha used at training time (scale = alpha / rank). Must match the trained lora_alpha.",
    )
    args = parser.parse_args()

    fuse_ckpt(args.ckpt_path, alpha=args.alpha)


if __name__ == "__main__":
    main()
