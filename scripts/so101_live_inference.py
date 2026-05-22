#!/usr/bin/env python3
"""Live robot inference for SO101 using the Video2World→World2Action pipeline.

Receives camera frames and joint states from a live robot, runs inference, and
returns predicted joint-position action sequences.

Run with the model venv:
    model/.venv/bin/python scripts/so101_live_inference.py \
        --experiment w2a_so101_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1 \
        --video_model_path /path/to/video_dit.pt \
        --action_model_path /path/to/action_dit.pt \
        --stats_path /path/to/dataset_statistics.json \
        --task "push the object to the right"
"""

import argparse
import collections
import json
import pathlib
import sys
import time
from pathlib import Path

import einops
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _add_model_to_path() -> None:
    model_dir = str(Path(__file__).resolve().parents[1] / "model")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)


_add_model_to_path()

from cosmos_predict2.configs.config import make_config  # noqa: E402
from cosmos_predict2.data.action.utils import extract_normalization_types  # noqa: E402
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline  # noqa: E402
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline  # noqa: E402
from cosmos_predict2.pipelines.world2action import World2ActionPipeline  # noqa: E402
from imaginaire.lazy_config import instantiate  # noqa: E402
from imaginaire.utils.config_helper import override  # noqa: E402


def load_pipeline(
    experiment: str,
    video_model_path: str,
    action_model_path: str,
    stats_path: pathlib.Path,
) -> Video2World2ActionPipeline:
    config = make_config()
    config = override(config, ["--", f"experiment={experiment}"])
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    video_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=video_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
    )
    action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=action_model_path,
        device="cuda",
        dtype=torch.bfloat16,
    )

    data_config = instantiate(config.data_config)
    with stats_path.open() as f:
        stats = json.load(f)

    action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return Video2World2ActionPipeline(video_pipe, action_pipe).cuda()


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    """Resize uint8 HWC image to 480×640 and normalize to [-1, 1], returning (3, 1, H, W)."""
    img = np.array(transforms.Resize((480, 640))(Image.fromarray(image, "RGB")))
    img = einops.rearrange(img, "h w c -> c h w")[:, None, :, :]
    return 2.0 * (img.astype(np.float32) / 255.0 - 0.5)


class SO101LiveInference:
    """Stateful inference wrapper for a live SO101 robot.

    Usage:
        infer = SO101LiveInference.from_checkpoint(...)
        infer.reset("push the object to the right")
        for step in episode:
            image = camera.capture()          # np.uint8 (H, W, 3)
            state = robot.joint_positions()   # np.float32 (6,)
            action = infer.step(image, state) # np.float32 (6,) joint positions
            robot.set_joint_positions(action)
    """

    def __init__(
        self,
        pipeline: Video2World2ActionPipeline,
        img_horizon: int = 5,
        lowdim_horizon: int = 5,
        num_sampling_steps: int = 35,
        stop_after_step: int | None = None,
        num_execute_actions: int = 1,
    ) -> None:
        self.model = pipeline
        self._img_horizon = img_horizon
        self._lowdim_horizon = lowdim_horizon
        self._num_sampling_steps = num_sampling_steps
        self._stop_after_step = stop_after_step
        self._num_execute_actions = num_execute_actions

        self._task: str = ""
        self._img_buf: collections.deque | None = None
        self._lowdim_buf: collections.deque | None = None
        self._action_buf: np.ndarray | None = None  # (HA, A)
        self._action_idx: int = 0

    @classmethod
    def from_checkpoint(
        cls,
        experiment: str,
        video_model_path: str,
        action_model_path: str,
        stats_path: pathlib.Path,
        **kwargs,
    ) -> "SO101LiveInference":
        pipeline = load_pipeline(experiment, video_model_path, action_model_path, stats_path)
        return cls(pipeline, **kwargs)

    def reset(self, task: str) -> None:
        self._task = task
        self._img_buf = None
        self._lowdim_buf = None
        self._action_buf = None
        self._action_idx = 0

    def step(self, image: np.ndarray, joint_state: np.ndarray) -> np.ndarray:
        """Run one inference step.

        Args:
            image: Camera frame, uint8 (H, W, 3).
            joint_state: Current joint positions, float32 (6,).

        Returns:
            Predicted joint positions to command, float32 (6,).
        """
        assert image.dtype == np.uint8, "image must be uint8"
        assert joint_state.shape == (6,), f"expected 6D joint state, got {joint_state.shape}"

        frame = _preprocess_image(image)  # (3, 1, 480, 640)

        if self._img_buf is None:
            self._img_buf = collections.deque(maxlen=self._img_horizon)
        self._img_buf.append(frame)

        if self._lowdim_buf is None:
            self._lowdim_buf = collections.deque(maxlen=self._lowdim_horizon)
            self._lowdim_buf.extend([joint_state.copy()] * self._lowdim_horizon)
        else:
            self._lowdim_buf.append(joint_state.copy())

        if self._action_buf is None:
            self._action_buf = self._run_inference()
            self._action_idx = 0

        action = self._action_buf[self._action_idx].copy()
        self._action_idx += 1
        if self._action_idx >= self._num_execute_actions:
            self._action_buf = None

        return action

    def _run_inference(self) -> np.ndarray:
        if len(self._img_buf) == self._img_horizon:
            imgs = np.concatenate(list(self._img_buf), axis=1)  # (3, T, 480, 640)
        else:
            # pad with first frame if history not yet full
            imgs = np.repeat(self._img_buf[-1], self._img_horizon, axis=1)

        lowdims = np.stack(list(self._lowdim_buf), axis=0)  # (HO, 6)

        input_vid = torch.from_numpy(imgs[None]).cuda().bfloat16()   # (1, 3, T, 480, 640)
        state = torch.from_numpy(lowdims[None]).cuda().bfloat16()    # (1, HO, 6)

        pred = self.model(
            input_vid=input_vid,
            state_B_HO_O=state,
            prompt=self._task,
            num_sampling_step=self._num_sampling_steps,
            stop_after_step=self._stop_after_step,
        )
        # pred: (1, action_horizon, 6)
        return pred[0].float().cpu().numpy()  # (action_horizon, 6)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO101 live robot inference")
    p.add_argument(
        "--experiment",
        default="w2a_so101_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1",
        help="Experiment config name (action decoder experiment)",
    )
    p.add_argument("--video_model_path", required=True, help="Path to video DiT checkpoint")
    p.add_argument("--action_model_path", required=True, help="Path to action DiT checkpoint")
    p.add_argument("--stats_path", required=True, type=Path, help="Path to dataset_statistics.json")
    p.add_argument("--task", default="push the object to the right", help="Task description")
    p.add_argument("--img_horizon", type=int, default=5)
    p.add_argument("--lowdim_horizon", type=int, default=5)
    p.add_argument("--num_sampling_steps", type=int, default=35)
    p.add_argument(
        "--stop_after_step",
        type=int,
        default=None,
        help="V2W denoising step at which to extract context (None = full rollout)",
    )
    p.add_argument("--num_execute_actions", type=int, default=1, help="Actions to execute before recomputing")
    p.add_argument("--demo_steps", type=int, default=5, help="Number of demo steps to run with synthetic data")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Loading pipeline (experiment={args.experiment}) ...", flush=True)
    infer = SO101LiveInference.from_checkpoint(
        experiment=args.experiment,
        video_model_path=args.video_model_path,
        action_model_path=args.action_model_path,
        stats_path=args.stats_path,
        img_horizon=args.img_horizon,
        lowdim_horizon=args.lowdim_horizon,
        num_sampling_steps=args.num_sampling_steps,
        stop_after_step=args.stop_after_step,
        num_execute_actions=args.num_execute_actions,
    )
    print("Pipeline loaded.", flush=True)

    print(f"\nRunning demo with {args.demo_steps} synthetic steps.")
    print(f"Task: {args.task!r}\n")

    infer.reset(args.task)
    rng = np.random.default_rng(42)

    for step_i in range(args.demo_steps):
        # Synthetic robot data — replace with real camera/robot reads.
        image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        joint_state = rng.uniform(-1.0, 1.0, (6,)).astype(np.float32)

        t0 = time.perf_counter()
        action = infer.step(image, joint_state)
        dt = time.perf_counter() - t0

        print(f"step {step_i:3d} | inference {dt*1000:.0f} ms | action {np.array2string(action, precision=4)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
