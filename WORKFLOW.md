1. Activate the project venv
2. Use the script in `utils` to convert the LeRobot-style dataset to the expected ZARR format.
```bash
uv run ./data_preprocessing/action/process_lerobot.py
    --dataset-path <path_to_lerobot_v3_dataset> \
    --output-dir <path_to_output_zarr_folder> \
    [--num-workers <int>] \
    [--fps <int>] \
    [--episodes <list_of_indices>] \
    [--overwrite]
```
For example:
```bash
uv run ./data_preprocessing/action/process_lerobot.py \
  --dataset-path ~/.cache/huggingface/hub/datasets--ETHRC--robot-learning-fs26/snapshots/a49199473f399e962f26777c2ef89d9e19f1a20e \
  --output-dir ~/zarr \
  --num-workers 24
```
3. Precompute language embeddings
```bash
uv run ../data_preprocessing/action/precompute_t5.py --dataset-path <path_to_lerobot_v3_dataset>
```
For example:
```bash
uv run ../data_preprocessing/action/precompute_t5.py --dataset-path ~/.cache/huggingface/hub/datasets--ETHRC--robot-learning-fs26/snapshots/a49199473f399e962f26777c2ef89d9e19f1a20e
```
