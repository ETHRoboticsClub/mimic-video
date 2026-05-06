# Steps mimic video EVAL

# Create machine with enough GPUs (tested with 8xA100 40gb), preferably on us-east-1 
ec2 ami type prefered: 'Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) 20260222'

# make sure enough space, only the checkpoints are taking up to 100GB

# mimic video repo
git clone https://github.com/mimic-video/mimic-video.git
cd mimic-video/model
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --extra cu126
source .venv/bin/activate

# download checkpoints
aws login --remote
aws s3 sync s3://ethrc-ml-data-916780037007/mimic/ . --region us-east-1 

# eval
cd ../eval/libero
uv pip install -r LIBERO/requirements.txt
uv pip install -e LIBERO

# modify eval.sh: 
# set checkpoint dir (should be under .../mimic-video/model/checkpoints)
# -adjust GPUs  (depending on num of gpus) 
# -optionally select the relevant task under models=(....)
# MODIFY line 8 for i in $(seq 0 $(( 1 * ${#GPUS[@]} - 1 ))); do
  g=$((i % ${#GPUS[@]}))
  echo "$g" >&3
done
bash eval.sh

