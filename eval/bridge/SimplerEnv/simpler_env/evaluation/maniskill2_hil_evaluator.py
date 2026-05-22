"""
Human-in-the-loop evaluator for ManiSkill2 + SIMPLER (single-loop per episode).
Loop: save_state -> teleop -> restore_state -> policy rollout (K steps) -> repeat.
Saves teleop videos under hil_results/ with the same inner structure as results/.
Skips an episode if any existing .mp4 in that directory contains the episode key.
"""

import gc
import os
import pathlib
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from term_image import disable_queries
from term_image.image import AutoImage

from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video

# --------------------------- Teleop utilities ---------------------------

disable_queries()


@dataclass
class TeleopConfig:
    lin_step: float = 0.015
    rot_step: float = np.deg2rad(30.0)
    fps: int = 5


class _KeyState:
    def __init__(self):
        self.down = set()
        self.stop_segment = False

    def on_press(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = str(key)
        if k == "c":
            self.stop_segment = True
        else:
            self.down.add(k)

    def on_release(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = str(key)
        self.down.discard(k)


class _StdinKeyReader:
    """
    Headless key reader for SSH/TTY. No X requirement.
    Holds a key "down" briefly so repeated chars act like a hold.
    """

    def __init__(self, key_state: _KeyState, ttl: float = 0.08):
        self.ks = key_state
        self.ttl = ttl
        self._stop = False
        self._last = {}

    def start(self):
        self._stream = sys.stdin if sys.stdin.isatty() else open("/dev/tty")
        self._fd = self._stream.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=0.2)
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        if self._stream is not sys.stdin:
            try:
                self._stream.close()
            except Exception:
                pass

    def _run(self):
        import time

        in_esc = False  # inside an ESC-initiated control/CSI sequence
        in_osc = False  # inside an OSC (Operating System Command) sequence
        prev_was_esc = False

        while not self._stop:
            r, _, _ = select.select([self._fd], [], [], 0.02)
            now = time.time()
            if r:
                try:
                    ch = os.read(self._fd, 1)
                except Exception:
                    ch = b""

                if ch:
                    b = ch[0]

                    # Handle OSC termination via BEL or ESC \
                    if in_osc:
                        if b == 7:  # BEL
                            in_osc = False
                            prev_was_esc = False
                            continue
                        if prev_was_esc and b == 0x5C:  # ESC \
                            in_osc = False
                            prev_was_esc = False
                            continue
                        prev_was_esc = b == 0x1B
                        continue

                    # Inside ESC-initiated (CSI/SS3) sequence: consume until final byte @..~
                    if in_esc:
                        if b == ord("]"):
                            # ESC ] ...  => switch to OSC parsing
                            in_esc = False
                            in_osc = True
                            prev_was_esc = False
                            continue
                        if 64 <= b <= 126:  # final byte of CSI/SS3/etc
                            in_esc = False
                            prev_was_esc = False
                            continue
                        # still inside the sequence
                        prev_was_esc = b == 0x1B
                        continue

                    # Start of an escape/control sequence
                    if b == 0x1B:  # ESC
                        in_esc = True
                        prev_was_esc = True
                        continue

                    # Regular printable byte -> map to command keys
                    c = chr(b).lower()
                    if c == "c":
                        self.ks.stop_segment = True
                    else:
                        self.ks.down.add(c)
                        self._last[c] = now

            # Expire "held" keys if not repeated
            for k, t in list(self._last.items()):
                if now - t > self.ttl:
                    self._last.pop(k, None)
                    self.ks.down.discard(k)


def _keyboard_to_action(keys: set, grip_target: float, cfg: TeleopConfig) -> tuple[dict[str, np.ndarray], float, bool]:
    move = np.zeros(3, dtype=np.float32)
    rot = np.zeros(3, dtype=np.float32)

    if "w" in keys:
        move[0] += cfg.lin_step
    if "s" in keys:
        move[0] -= cfg.lin_step
    if "a" in keys:
        move[1] += cfg.lin_step
    if "d" in keys:
        move[1] -= cfg.lin_step
    if "r" in keys:
        move[2] += cfg.lin_step
    if "f" in keys:
        move[2] -= cfg.lin_step

    if "i" in keys:
        rot[0] += cfg.rot_step
    if "k" in keys:
        rot[0] -= cfg.rot_step
    if "j" in keys:
        rot[1] += cfg.rot_step
    if "l" in keys:
        rot[1] -= cfg.rot_step
    if "u" in keys:
        rot[2] += cfg.rot_step
    if "o" in keys:
        rot[2] -= cfg.rot_step

    if "z" in keys:
        grip_target = -1.0
    if "x" in keys:
        grip_target = 1.0

    wait = " " in keys

    return (
        {
            "world_vector": move,
            "rot_axangle": rot,
            "gripper": np.array([grip_target], dtype=np.float32),
            "terminate_episode": np.array([0.0], dtype=np.float32),
        },
        grip_target,
        wait,
    )


def _termimage_preview(img: np.ndarray) -> None:
    ti = AutoImage(Image.fromarray(img, mode="RGB").resize((140, 105)), width=140)
    print("\033[H\033[2J", end="")
    ti.draw()
    print("\n[C=end teleop]", end="", flush=True)


def _print_teleop_status(
    keys: set, grip: float, step_count: int, target_pos: np.ndarray, obj_pos: np.ndarray, eef_pos: np.ndarray
) -> None:
    print(f"\n| grip target: {grip:.2f} | {target_pos=} | {obj_pos=} | {eef_pos=}", end="", flush=True)


def _policy_step_action(env, model, image, task_description):
    ctrl = env.agent.controllers["arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"]
    ee_pose_proprio = ctrl.controllers["arm"].ee_pose_at_base
    g = ctrl.controllers["gripper"].qpos.mean()
    g_norm = (g - 0.015) / (0.037 - 0.015)
    act = model.step(image, task_description, ee_pose_proprio, g_norm)
    return np.concatenate([act["world_vector"], act["rot_axangle"], act["gripper"]])


def _teleop_segment(env, grip_target, key_state: _KeyState, cfg: TeleopConfig, obs_camera_name: str | None) -> dict:
    print("\n\n ================ get ready ================ \n\n")
    time.sleep(2)

    images = []
    last_grip_cmd = grip_target
    done = False
    truncated = False
    info = {}
    step_count = 0
    EPS = 1e-8

    controller = env.agent.controllers["arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"]
    controller.controllers["arm"].reset()
    obs, reward, done, truncated, info = env.step(np.array([0, 0, 0, 0, 0, 0, grip_target]))

    img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    _termimage_preview(img)
    _print_teleop_status(
        key_state.down,
        (
            env.agent.controllers["arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"]
            .controllers["gripper"]
            ._target_qpos.mean()
            - 0.015
        )
        / (0.037 - 0.015),
        step_count,
        (env.target_obj_pose.p * 100).tolist(),
        (env.source_obj_pose.p * 100).tolist(),
        (
            env.agent.controllers["arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"].controllers["arm"].ee_pos
            * 100
        ).tolist(),
    )

    while True:
        action_dict, grip_target, wait = _keyboard_to_action(key_state.down, grip_target, cfg)

        world = action_dict["world_vector"]
        rot = action_dict["rot_axangle"]
        grip_arr = action_dict["gripper"].ravel()
        grip_cmd = float(grip_arr[0]) if grip_arr.size else float(last_grip_cmd)

        is_noop = (
            not wait
            and np.all(np.abs(world) <= EPS)
            and np.all(np.abs(rot) <= EPS)
            and abs(grip_cmd - last_grip_cmd) <= EPS
        )

        if is_noop:
            if key_state.stop_segment:
                key_state.stop_segment = False
                break
            time.sleep(0.01)
            continue

        action = np.concatenate([world, rot, grip_arr])
        obs, reward, done, truncated, info = env.step(action)
        img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)

        step_count += 1
        if step_count > 1:
            images.append(img)

        _termimage_preview(img)
        _print_teleop_status(
            key_state.down,
            float(action_dict["gripper"][0]),
            step_count,
            (env.target_obj_pose.p * 100).tolist(),
            (env.source_obj_pose.p * 100).tolist(),
            (
                env.agent.controllers["arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"]
                .controllers["arm"]
                .ee_pos
                * 100
            ).tolist(),
        )

        last_grip_cmd = grip_cmd

        if key_state.stop_segment:
            key_state.stop_segment = False
            break

    return {
        "images": images or [img],
        "done": bool(done),
        "truncated": bool(truncated),
        "info": info,
        "steps": step_count,
    }


# --------------------------- Single-episode runner ---------------------------


def run_maniskill2_hil_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=200,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./hil_results",
):
    additional_env_build_kwargs = {} if additional_env_build_kwargs is None else dict(additional_env_build_kwargs)

    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]

    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"

    r, p, y = Rotation.from_quat(robot_init_quat, scalar_first=True).as_euler(seq="XYZ")

    # hil_results/ next to results/
    assert "hil_results" in logging_dir

    subdir = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}"
    video_dir = os.path.join(logging_dir, subdir)

    if os.path.exists(video_dir):
        for mp4 in pathlib.Path(video_dir).iterdir():
            if not mp4.is_file() or mp4.suffix != ".mp4":
                continue
            key_xy = f"obj_{obj_init_x}_{obj_init_y}" if (obj_init_x is not None and obj_init_y is not None) else None
            key_ep = f"obj_episode_{obj_episode_id}" if obj_episode_id is not None else None
            if (key_xy and key_xy in mp4.name) or (key_ep and key_ep in mp4.name):
                print(
                    f"Skipping (HIL) {mp4.parents[1].name}/{obj_episode_id or (str(obj_init_x) + '_' + str(obj_init_y))}, already done."
                )
                return "success" in mp4.name

    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        rt = {"shader_dir": "rt"}
        rt.update(additional_env_build_kwargs)
        additional_env_build_kwargs = rt
    env = build_maniskill2_env(env_name, **additional_env_build_kwargs, **kwargs)

    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None and obj_init_y is not None:
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {"init_xy": np.array([obj_init_x, obj_init_y])}
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {"episode_id": obj_episode_id}
    obs, _ = env.reset(options=env_reset_options)

    task_description = instruction if instruction is not None else env.get_language_instruction()
    model.reset(task_description)

    cfg = TeleopConfig()
    ks = _KeyState()
    reader = _StdinKeyReader(ks)
    reader.start()

    seg_idx = 0
    print("[Teleop] W/S/A/D XY, R/F Z, I/K/J/L/U/O rot, Z/X grip, C end teleop.")
    timestep = 0
    truncated = False
    policy_images: list[np.ndarray] = [get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)]
    last_info: dict = {}
    gripper_target = 1.0

    while not truncated:
        S = env.get_state()
        target_arm = (
            env.agent.controllers["arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"]
            .controllers["arm"]
            ._target_pose
        )
        gripper_controller_state = (
            env.agent.controllers["arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"]
            .controllers["gripper"]
            .get_state()
        )

        seg = _teleop_segment(env, gripper_target, ks, cfg, obs_camera_name)
        hil_images = seg["images"]

        env.set_state(S)
        env.agent.controllers["arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"].controllers[
            "arm"
        ]._target_pose = target_arm
        env.agent.controllers["arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"].controllers[
            "gripper"
        ].set_state(gripper_controller_state)

        model.ingest_video(hil_images)

        img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        for _ in range(model.num_execute_actions):
            act_vec = _policy_step_action(env, model, img, task_description)
            obs, _reward, done, _truncated, info = env.step(act_vec)
            truncated = timestep == max_episode_steps - 1
            print(timestep, info)

            if truncated:
                img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
                policy_images.append(img)
                last_info = info
                break
            new_desc = env.get_language_instruction()
            if new_desc != task_description:
                task_description = new_desc
            img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
            policy_images.append(img)
            timestep += 1
        seg_idx += 1
        gripper_target = act_vec[-1]

    reader.stop()
    env.close()
    del env
    gc.collect()

    success = "success" if done else "failure"
    episode_stats = last_info.get("episode_stats", {})
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    else:
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    result_video_path = os.path.join(video_dir, video_name + ".mp4")
    if policy_images:
        write_video(result_video_path, policy_images, fps=5)
        print(f"[Saved rollout video] {result_video_path}")
    action_dir = os.path.join(video_dir, "actions")
    os.makedirs(action_dir, exist_ok=True)
    action_path = os.path.join(action_dir, video_name + ".png")
    model.visualize_epoch(policy_images, save_path=action_path)

    return done


def maniskill2_hil_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot)
    success_arr = []

    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                common = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_hil_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **common,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(
                            run_maniskill2_hil_single_episode(
                                obj_episode_id=obj_episode_id,
                                **common,
                            )
                        )
                else:
                    raise NotImplementedError()

    return success_arr
