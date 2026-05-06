# Docker Setup for Local Inference

This directory contains Docker configuration for running mimic-video inference locally using NVIDIA's Cosmos Predict2 container.

## Prerequisites

- Docker Desktop or Docker Engine with NVIDIA Container Toolkit installed
- NVIDIA GPU with CUDA 12.6+ support
- Hugging Face account (for downloading checkpoints)

### Install NVIDIA Container Toolkit

For Ubuntu/Debian:
```bash
# Add Docker's repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -s https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb #deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] #g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

For other platforms, see: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## Setup

### 1. Create Required Directories

```bash
mkdir -p checkpoints output data
```

### 2. Download Checkpoints

You need to download the model checkpoints to the `checkpoints/` directory:

```bash
# Set your Hugging Face token (required for accessing some models)
export HF_TOKEN=your_token_here

# Download checkpoints using the provided script
cd model
python scripts/download_checkpoints.py --output-dir ../checkpoints
```

If you want to use guardrails (enabled by default), also download:
- `nvidia/Cosmos-Guardrail1`
- `meta-llama/Llama-Guard-3-8B`

### 3. Configure Environment (Optional)

Create a `.env.local` file to store sensitive configuration:

```bash
# Copy the example
cp .env.example .env.local

# Edit with your values
vim .env.local  # or use any editor
```

Set the following variables as needed:
- `HF_TOKEN`: Your Hugging Face access token
- `CUDA_VISIBLE_DEVICES`: GPU ID(s) to use (default: 0)

## Usage

### Start the Container

```bash
# Start in interactive mode (ready for inference commands)
docker-compose up

# Or start in detached mode
docker-compose up -d

# Start and run a specific command
docker-compose run --rm mimic-video-inference \
  python scripts/run_video2world.py --help
```

### Run Inference

Inside the container (or via `docker-compose run`):

```bash
# Basic video-to-world generation
python scripts/run_video2world.py \
  --dit_path /workspace/checkpoints/v2w_pretrained_cosmos \
  --prompt "robot arm picks up the block" \
  --input_path /workspace/data/input.mp4 \
  --save_path /workspace/output/generated.mp4

# With custom guidance and seed
python scripts/run_video2world.py \
  --dit_path /workspace/checkpoints/v2w_pretrained_cosmos \
  --prompt "robot places the object on the shelf" \
  --input_path /workspace/data/input.mp4 \
  --save_path /workspace/output/generated.mp4 \
  --guidance 7.5 \
  --seed 42 \
  --num_conditional_frames 1

# Disable guardrails (faster, but no safety checks)
python scripts/run_video2world.py \
  --dit_path /workspace/checkpoints/v2w_pretrained_cosmos \
  --prompt "..." \
  --input_path /workspace/data/input.mp4 \
  --save_path /workspace/output/generated.mp4 \
  --disable_guardrail

# Memory-efficient mode (offload models to CPU when not in use)
python scripts/run_video2world.py \
  --dit_path /workspace/checkpoints/v2w_pretrained_cosmos \
  --prompt "..." \
  --input_path /workspace/data/input.mp4 \
  --save_path /workspace/output/generated.mp4 \
  --offload_text_encoder \
  --offload_guardrail
```

### Batch Processing

Create a JSON file with batch inputs:

```json
[
  {
    "input_video": "/workspace/data/video1.mp4",
    "prompt": "robot picks up block A",
    "output_video": "/workspace/output/result1.mp4"
  },
  {
    "input_video": "/workspace/data/video2.mp4",
    "prompt": "robot places block on table",
    "output_video": "/workspace/output/result2.mp4"
  }
]
```

Run batch inference:

```bash
python scripts/run_video2world.py \
  --dit_path /workspace/checkpoints/v2w_pretrained_cosmos \
  --batch_input_json /workspace/data/batch_inputs.json \
  --disable_guardrail
```

## Directory Structure

```
mimic-video/
├── docker-compose.yml      # Docker compose configuration
├── .env.example            # Example environment variables
├── checkpoints/            # Downloaded model checkpoints (create this)
│   ├── v2w_pretrained_cosmos/
│   ├── bridge-action-decoder/
│   └── libero-action-decoder/
├── data/                   # Input videos/images (create this)
│   └── input.mp4
├── output/                 # Generated videos (create this)
│   └── generated.mp4
└── model/                  # Model code (already exists)
    └── scripts/
        └── run_video2world.py
```

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA Container Toolkit is working
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

If this fails, reinstall the NVIDIA Container Toolkit.

### Out of Memory

Use memory offloading options:
- `--offload_text_encoder`: Move text encoder to CPU
- `--offload_guardrail`: Move guardrail models to CPU
- Reduce `--num_conditional_frames` from 5 to 1

### Checkpoint Download Fails

Ensure `HF_TOKEN` is set and you have access to the required repositories:

```bash
export HF_TOKEN=your_token_here
huggingface-cli login
```

### Permission Issues

If you encounter permission errors with mounted volumes, try:

```bash
# Reset permissions on local directories
sudo chmod -R 755 checkpoints data output
```

## Performance Tips

1. **Use NVMe storage** for checkpoints and data directories
2. **Pre-load checkpoints** into container before running inference
3. **Use multi-GPU** for batch processing (set `CUDA_VISIBLE_DEVICES=0,1`)
4. **Disable guardrails** for faster inference in trusted environments
5. **Use CUDA graphs** (`--use_cuda_graphs`) for optimized performance

## Alternative Container Options

While this setup uses NVIDIA's official Cosmos Predict2 container, you could alternatively:

1. **Build a custom container** with all dependencies pre-installed
2. **Use a base PyTorch container** and install dependencies manually
3. **Use Singularity/Apptainer** for HPC environments

Contact the maintainers if you need assistance with alternative setups.
