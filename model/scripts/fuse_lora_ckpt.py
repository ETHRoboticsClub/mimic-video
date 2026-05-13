import argparse

import torch


def fuse_ckpt(ckpt_path: str, alpha: int) -> str:
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

        adapter = ckpt[b_key] @ ckpt[key]
        fused = ckpt[base_key] + alpha / this_rank * adapter

        del ckpt[key], ckpt[b_key], ckpt[base_key]

        ckpt[base_key.replace(".base_layer", "")] = fused

    print(f"{lora_rank=} alpha={alpha} scale={alpha/lora_rank if lora_rank else 'n/a'}")

    fused_path = ckpt_path.replace(".pt", "_fused.pt")
    torch.save(ckpt, fused_path)

    return fused_path


def main():
    parser = argparse.ArgumentParser(description="Fuse LoRA adapters into base weights.")
    parser.add_argument("ckpt_path", type=str, help="checkpoint path")
    parser.add_argument(
        "--alpha",
        type=int,
        default=32,
        help=(
            "LoRA alpha used at training time. Default 32 matches upstream bridge/libero_per-suite "
            "checkpoints. ETHRC unified libero cosmos finetune used alpha=16 (verified from "
            "s3://.../2b_libero_cosmos/config.yaml). If wrong alpha is passed, fused weights "
            "will be silently mis-scaled."
        ),
    )
    args = parser.parse_args()

    fuse_ckpt(args.ckpt_path, args.alpha)


if __name__ == "__main__":
    main()
