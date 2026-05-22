from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import threading

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from decord import VideoReader, cpu

from cosmos_predict2.data.dataset_video import _stable_hash_int
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
from imaginaire.utils import distributed, log
from imaginaire.utils.callback import Callback
from imaginaire.utils.io import save_image_or_video

_VAL_HASH_DENOM = 10_000


class VideoEvalCallback(Callback):
    def __init__(
        self,
        fuse_lora: bool,
        lora_alpha: float = 32,
        enabled: bool = False,
        eval_video_dir: str | None = None,
        prompt: str | None = None,
        val_ratio: float = 0.0,
        hf_repo_id: str | None = None,
        episode_stem: str | None = None,
        episode_index: int | None = None,
        max_episodes_per_val_cycle: int | None = None,
        seed_start_frame: int = 0,
        num_seed_frames: int = 5,
        num_eval_video_frames: int | None = None,
        seed: int = 0,
        guidance: float = 7.0,
        num_sampling_step: int = 35,
    ):
        self._fuse_lora = fuse_lora
        self._lora_alpha = lora_alpha
        self._enabled = enabled
        self._eval_video_dir = eval_video_dir
        self._prompt = prompt or ""
        self._val_ratio = val_ratio
        self._hf_repo_id = hf_repo_id
        self._episode_stem = episode_stem
        self._episode_index = episode_index
        self._max_episodes_per_val_cycle = max_episodes_per_val_cycle
        self._seed_start_frame = seed_start_frame
        self._num_seed_frames = num_seed_frames
        self._num_eval_video_frames = num_eval_video_frames
        self._seed = seed
        self._guidance = guidance
        self._num_sampling_step = num_sampling_step
        self._fuse_proc: subprocess.Popen | None = None
        self._hf_thread: threading.Thread | None = None

    @distributed.rank0_only
    def on_save_checkpoint_success(
        self, iteration: int = 0, elapsed_time: float = 0, checkpoint_path: str | None = None
    ) -> None:
        if not (self._fuse_lora or self._hf_repo_id):
            return
        # Checkpointer.save() invokes this callback once per saved subfolder
        # (model, optim, scheduler, trainer). Only act on the model file so the
        # fuse and HF-upload work runs exactly once per save iter.
        if checkpoint_path is None or os.path.basename(os.path.dirname(checkpoint_path)) != "model":
            return

        if self._fuse_lora:
            self._launch_fuse_subprocess(iteration, checkpoint_path)

        if self._hf_repo_id:
            # HF upload is small and network-bound; keep it on a background
            # thread so it doesn't block the trainer.
            if self._hf_thread is not None and self._hf_thread.is_alive():
                log.info("video_eval: previous HF upload still running; joining before launching new one.")
                self._hf_thread.join()
            self._hf_thread = threading.Thread(
                target=self._upload_adapter_to_hf,
                kwargs={"iteration": iteration},
                name=f"video_eval_hf_upload_iter{iteration:09d}",
                daemon=False,
            )
            self._hf_thread.start()

    def _launch_fuse_subprocess(self, iteration: int, checkpoint_path: str) -> None:
        # Fuse used to run as an in-process thread, but loading + matmuls +
        # 5+ GB torch.save inside the training process triggered OOM-kills and
        # blocked the trainer when joined at the next save. An isolated
        # subprocess: (a) frees its memory immediately on exit, (b) can be
        # niced so it can't starve training, (c) lets us skip-if-busy without
        # joining the trainer on a slow disk write.
        if self._fuse_proc is not None and self._fuse_proc.poll() is None:
            log.warning(
                f"video_eval: previous fuse subprocess (pid {self._fuse_proc.pid}) still running; "
                f"skipping fuse for iter {iteration} to avoid pile-up. Re-fuse offline if needed."
            )
            return
        base_cmd = [
            sys.executable,
            "-m",
            "scripts.fuse_lora_ckpt",
            str(checkpoint_path),
            "--alpha",
            str(self._lora_alpha),
        ]
        # Prepend `nice -n 19 ionice -c 3` if available for lowest CPU/IO priority.
        wrappers: list[str] = []
        if shutil.which("nice"):
            wrappers += ["nice", "-n", "19"]
        if shutil.which("ionice"):
            wrappers += ["ionice", "-c", "3"]
        cmd = wrappers + base_cmd
        try:
            self._fuse_proc = subprocess.Popen(
                cmd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info(
                f"video_eval: spawned fuse subprocess pid {self._fuse_proc.pid} for iter {iteration}."
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(f"video_eval: failed to spawn fuse subprocess for iter {iteration}: {exc}")
            self._fuse_proc = None

    def _upload_adapter_to_hf(self, iteration: int) -> None:
        from huggingface_hub import HfApi, create_repo

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        api = HfApi(token=token)
        try:
            create_repo(
                repo_id=self._hf_repo_id,
                repo_type="model",
                private=True,
                exist_ok=True,
                token=token,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(f"video_eval: failed to ensure HF repo {self._hf_repo_id}: {exc}")
            return

        job_name = self.config.job.name
        checkpoints_root = os.path.join(self.config.job.path_local, "checkpoints")
        iter_filename = f"iter_{iteration:09d}.pt"
        model_path = os.path.join(checkpoints_root, "model", iter_filename)
        if not os.path.exists(model_path):
            log.warning(f"video_eval: model checkpoint not found at {model_path}; skipping HF upload.")
            return

        state = torch.load(model_path, map_location="cpu", weights_only=True)
        adapter_state = {
            k: v for k, v in state.items() if "lora" in k.lower() or "adaln_modulation" in k
        }
        if not adapter_state:
            log.warning(
                f"video_eval: no LoRA / adaln_modulation params found in {model_path}; "
                "nothing to upload."
            )
            return

        adapter_local = os.path.join(checkpoints_root, "lora_adapter", iter_filename)
        os.makedirs(os.path.dirname(adapter_local), exist_ok=True)
        torch.save(adapter_state, adapter_local)

        repo_path = f"{job_name}/lora_adapter/{iter_filename}"
        log.info(
            f"video_eval: uploading LoRA adapter ({len(adapter_state)} tensors) "
            f"for iter {iteration} to hf://{self._hf_repo_id}/{repo_path}."
        )
        try:
            api.upload_file(
                path_or_fileobj=adapter_local,
                path_in_repo=repo_path,
                repo_id=self._hf_repo_id,
                repo_type="model",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(f"video_eval: failed to upload {adapter_local} -> {repo_path}: {exc}")
            return

        latest_local = os.path.join(checkpoints_root, "latest_checkpoint.txt")
        if os.path.exists(latest_local):
            try:
                api.upload_file(
                    path_or_fileobj=latest_local,
                    path_in_repo=f"{job_name}/latest_checkpoint.txt",
                    repo_id=self._hf_repo_id,
                    repo_type="model",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(f"video_eval: failed to upload latest_checkpoint.txt: {exc}")

        log.success(
            f"video_eval: uploaded LoRA adapter iter {iteration} to hf://{self._hf_repo_id}/{repo_path}."
        )

    def on_train_start(self, model, iteration: int = 0) -> None:
        # Fire one eval pass right after process start. On a fresh launch this
        # captures baseline pretrained quality; on auto-resume it surfaces
        # current quality immediately instead of waiting for the next
        # validation_iter cycle.
        if self._enabled and distributed.is_rank0():
            log.info(f"video_eval: generating eval at train start (iter {iteration}).")
            self._generate_and_log(model, iteration)
        distributed.barrier()

    def on_validation_end(self, model, iteration: int = 0) -> None:
        if self._enabled and distributed.is_rank0():
            self._generate_and_log(model, iteration)
        distributed.barrier()

    def _episode_basename(self) -> str:
        if self._episode_stem:
            return self._episode_stem if self._episode_stem.endswith(".mp4") else f"{self._episode_stem}.mp4"
        if self._episode_index is None:
            raise ValueError(
                "video_eval.enabled=true requires training.video_finetune.eval_video.episode_stem "
                "or episode_index."
            )
        return f"episode_{self._episode_index:06d}.mp4"

    def _in_val_split(self, basename: str) -> bool:
        return _stable_hash_int(os.path.splitext(basename)[0]) % _VAL_HASH_DENOM < round(self._val_ratio * _VAL_HASH_DENOM)

    def _resolve_video_path(self, basename: str) -> str:
        if self._eval_video_dir is None:
            raise ValueError("video_eval.enabled=true requires eval_video_dir.")
        video_path = os.path.join(self._eval_video_dir, "video", basename)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Validation eval video does not exist: {video_path}")
        return video_path

    def _resolve_basenames(self) -> list[str]:
        if self._eval_video_dir is None:
            raise ValueError("video_eval.enabled=true requires eval_video_dir.")
        if self._episode_stem or self._episode_index is not None:
            basename = self._episode_basename()
            if not self._in_val_split(basename):
                raise ValueError(
                    f"{basename} is not in the validation split for val_ratio={self._val_ratio}. "
                    "Pick an episode whose filename belongs to the validation hash split."
                )
            return [basename]
        video_dir = os.path.join(self._eval_video_dir, "video")
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"video_eval.eval_video_dir/video does not exist: {video_dir}")
        if round(self._val_ratio * _VAL_HASH_DENOM) <= 0:
            log.warning(
                "video_eval: val_ratio is 0 and no episode_stem/episode_index provided; "
                "skipping eval generation."
            )
            return []
        basenames = sorted(
            b for b in os.listdir(video_dir) if b.endswith(".mp4") and self._in_val_split(b)
        )
        if self._max_episodes_per_val_cycle is None:
            return basenames
        # Multi-source mixes prefix basenames with `sourceNNN_`. Plain alphabetical
        # truncation can pick all clips from the first source; round-robin by
        # source prefix so each source gets at least one slot before any source
        # gets a second.
        groups: dict[str, list[str]] = {}
        for b in basenames:
            key = b.split("_", 1)[0] if b.startswith("source") and "_" in b else ""
            groups.setdefault(key, []).append(b)
        picked: list[str] = []
        group_keys = sorted(groups.keys())
        while len(picked) < self._max_episodes_per_val_cycle:
            progress = False
            for key in group_keys:
                if not groups[key]:
                    continue
                picked.append(groups[key].pop(0))
                progress = True
                if len(picked) >= self._max_episodes_per_val_cycle:
                    break
            if not progress:
                break
        return picked

    def _load_prompt_embedding(self, video_path: str) -> torch.Tensor:
        t5_path = os.path.join(
            self._eval_video_dir,
            "t5_xxl",
            os.path.basename(video_path).replace(".mp4", ".pickle"),
        )
        if not os.path.exists(t5_path):
            raise FileNotFoundError(f"Validation eval T5 embedding does not exist: {t5_path}")

        with open(t5_path, "rb") as f:
            embedding_raw = pickle.load(f)
        if not isinstance(embedding_raw, list) or len(embedding_raw) != 1:
            raise ValueError(f"Unexpected T5 embedding format in {t5_path}")

        embedding = embedding_raw[0]
        if not isinstance(embedding, np.ndarray) or embedding.ndim != 2:
            raise ValueError(f"Unexpected T5 embedding array shape in {t5_path}")
        if embedding.shape[0] < CosmosTextEncoderConfig.NUM_TOKENS:
            embedding = np.concatenate(
                [
                    embedding,
                    np.zeros(
                        (
                            CosmosTextEncoderConfig.NUM_TOKENS - embedding.shape[0],
                            CosmosTextEncoderConfig.EMBED_DIM,
                        ),
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            )
        embedding = torch.from_numpy(embedding).unsqueeze(0)
        return embedding

    def _load_seed_video(self, video_path: str, num_video_frames: int) -> torch.Tensor:
        if self._seed_start_frame < 0:
            raise ValueError("eval_video.seed_start_frame must be non-negative.")
        if self._num_seed_frames not in {1, 5}:
            raise ValueError("eval_video.num_seed_frames must be 1 or 5.")

        reader = VideoReader(video_path, ctx=cpu(0), num_threads=0)
        end_frame = self._seed_start_frame + self._num_seed_frames
        if len(reader) < end_frame:
            raise ValueError(
                f"{video_path} has {len(reader)} frames, "
                f"but seed window [{self._seed_start_frame}:{end_frame}] was requested."
            )

        frames = reader.get_batch(np.arange(self._seed_start_frame, end_frame)).asnumpy()
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        if video.shape[-2:] != (480, 640):
            video = F.interpolate(
                video.float(),
                size=(480, 640),
                mode="bilinear",
                align_corners=False,
            ).to(torch.uint8)

        padded = torch.empty(num_video_frames, 3, 480, 640, dtype=torch.uint8)
        padded[: self._num_seed_frames] = video
        padded[self._num_seed_frames :] = video[-1]
        return padded.permute(1, 0, 2, 3).unsqueeze(0)

    @torch.no_grad()
    def _generate_and_log(self, model, iteration: int) -> None:
        basenames = self._resolve_basenames()
        if not basenames:
            return

        pipe = model.pipe
        # Optionally override generation length. Pipe.config.state_t controls
        # the latent T sampled by generate_video; we set it for the duration of
        # the eval loop and restore afterward.
        train_state_t = pipe.config.state_t
        if self._num_eval_video_frames is not None:
            eval_state_t = pipe.tokenizer.get_latent_num_frames(self._num_eval_video_frames)
            if eval_state_t != train_state_t:
                log.info(
                    f"video_eval: overriding pipe.config.state_t {train_state_t} -> {eval_state_t} "
                    f"({self._num_eval_video_frames} pixel frames) for generation."
                )
            pipe.config.state_t = eval_state_t
        num_video_frames = pipe.tokenizer.get_pixel_num_frames(pipe.config.state_t)
        num_latent_conditional_frames = pipe.tokenizer.get_latent_num_frames(self._num_seed_frames)

        output_dir = os.path.join(self.config.job.path_local, "video_eval")
        os.makedirs(output_dir, exist_ok=True)

        # Release fragmented cached blocks before the eval allocations spike.
        # Without this, on near-full GPUs the cached allocator can stall while
        # repacking, sometimes for minutes — looks like a hang.
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        log.info(f"video_eval: generating {len(basenames)} validation video(s) at iter {iteration}.")
        try:
            for basename in basenames:
                video_path = self._resolve_video_path(basename)
                stem = basename[: -len(".mp4")]
                seed_video = self._load_seed_video(video_path, num_video_frames)
                prompt_embedding = self._load_prompt_embedding(video_path)

                stem_dir = os.path.join(output_dir, stem)
                os.makedirs(stem_dir, exist_ok=True)
                output_path = os.path.join(stem_dir, f"iter_{iteration:09d}.mp4")

                log.info(
                    f"Generating validation video for {stem} from {video_path} frames "
                    f"[{self._seed_start_frame}:{self._seed_start_frame + self._num_seed_frames}]"
                )
                video = pipe.generate_video(
                    vid_input=seed_video,
                    num_latent_conditional_frames=num_latent_conditional_frames,
                    prompt="",
                    prompt_embedding=prompt_embedding,
                    guidance=self._guidance,
                    num_sampling_step=self._num_sampling_step,
                    seed=self._seed,
                )
                save_image_or_video(video, output_path, fps=10)
                log.success(f"Saved validation video eval: {output_path}")

                if wandb.run is not None:
                    wandb.log(
                        {
                            f"video_eval/{stem}": wandb.Video(output_path, fps=10, format="mp4"),
                            f"video_eval/{stem}/seed_start_frame": self._seed_start_frame,
                        },
                        step=iteration,
                    )
        finally:
            pipe.config.state_t = train_state_t
