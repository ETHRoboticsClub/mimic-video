# Cosmos Video Fine-tuning on Brev

LoRA fine-tune of the Cosmos 2B video backbone on SO-101 robot data.

## Prerequisites

- Brev instance with H100
- HuggingFace token with access to `rl26-world-models/*`
- Weights & Biases API key

## Setup (once per instance)

```bash
git clone <repo> && cd mimic-video
bash brev/setup_video_finetune.sh
```

Installs CUDA deps, creates Python venv via `uv`, logs into HF + W&B, downloads the Cosmos checkpoint (~10 GB).

## Run

**Smoke test** — preprocessing only, no actual training:
```bash
bash brev/run_video_finetune_smoke.sh
```

**Full training:**
```bash
tmux new -s train
bash brev/run_video_finetune.sh
# detach: Ctrl-B D  |  reattach: tmux attach
```

Closing the laptop lid is fine — tmux keeps the session alive over SSH.

## What the pipeline does

1. Downloads dataset from HuggingFace, resamples to 10 fps / 480×640
2. Encodes T5 text embeddings
3. Launches LoRA fine-tuning (rank 256, bfloat16)

Preprocessed videos land in `data/rl26-world-models/so101-task1-mini-v1-birdview/1/video_finetune/video/`.
Config: `configs/so101/video_finetune.yaml`.
