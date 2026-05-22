REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_ROOT}"
source .env

cd model
source .venv/bin/activate

python scripts/run_video2world.py \
  --dit_path "${REPO_ROOT}/model/checkpoints/video_backbone/v2w_pretrained_cosmos.pt" \
  --input_path "${REPO_ROOT}/data/sample_images/frame_0.png" \
  --prompt "the robot arm gently nudges the white object from the right into the circle" \
  --save_path "${REPO_ROOT}/data/output/cosmos_sample.mp4" \
  --disable_guardrail
