# original 1: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/regenerate_libero_dataset.py
# original 2: https://github.com/moojink/rlds_dataset_builder/blob/6174b0b6bb69df6361f1117944952bf14afb0cc3/LIBERO_10/LIBERO_10_dataset_builder.py

# Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

# Notes:
#     - Save video at higher resolution
#     - Filter out transitions with "no-op" (zero) actions that do not change the robot's state.
#     - Filter out unsuccessful demonstrations.
#     - Rotate images by 180 degrees because the environments returns images that are upside down.

import argparse
import multiprocessing as mp
import os
import pathlib
from functools import partial

import h5py
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640


def get_libero_env(task):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": CAMERA_HEIGHT, "camera_widths": CAMERA_WIDTH}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def process_one(task_id: int, suite: str, libero_raw_data_dir: pathlib.Path, libero_target_dir: pathlib.Path) -> None:
    num_replays = 0
    num_success = 0
    num_noops = 0

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()

    task = task_suite.get_task(task_id)
    env, task_description = get_libero_env(task)

    orig_data_path = libero_raw_data_dir / f"{task.name}_demo.hdf5"
    assert orig_data_path.exists(), f"Cannot find raw data file {orig_data_path}."
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]

    new_data_path = libero_target_dir / f"{task.name}_demo.hdf5"
    new_data_file = h5py.File(new_data_path, "w")
    grp = new_data_file.create_group("data")

    for i in range(len(orig_data.keys())):
        demo_data = orig_data[f"demo_{i}"]
        orig_actions = demo_data["actions"][()]
        orig_states = demo_data["states"][()]

        # Wait a few steps for environment to settle
        env.reset()
        env.set_init_state(orig_states[0])
        for _ in range(10):
            obs, _reward, done, _info = env.step(DUMMY_ACTION)

        states = []
        actions = []
        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []
        agentview_images = []
        eye_in_hand_images = []

        for _, action in enumerate(orig_actions):
            prev_action = actions[-1] if len(actions) > 0 else None
            if is_noop(action, prev_action):
                num_noops += 1
                continue

            if states == []:
                # In the first timestep, since we're using the original initial state to initialize the environment,
                # copy the initial state (first state in episode) over from the original HDF5 to the new one
                states.append(orig_states[0])
                robot_states.append(demo_data["robot_states"][0])
            else:
                states.append(env.sim.get_state().flatten())
                robot_states.append(
                    np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                )

            actions.append(action)

            if "robot0_gripper_qpos" in obs:
                gripper_states.append(obs["robot0_gripper_qpos"])
            joint_states.append(obs["robot0_joint_pos"])
            ee_states.append(
                np.hstack(
                    (
                        obs["robot0_eef_pos"],
                        obs["robot0_eef_quat"],
                    )
                )
            )
            agentview_images.append(obs["agentview_image"][::-1, ::-1])
            eye_in_hand_images.append(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            obs, _reward, done, _info = env.step(action.tolist())

        if done:
            dones = np.zeros(len(actions)).astype(np.uint8)
            dones[-1] = 1
            rewards = np.zeros(len(actions)).astype(np.uint8)
            rewards[-1] = 1
            assert len(actions) == len(agentview_images)

            ep_data_grp = grp.create_group(f"demo_{i}")
            obs_grp = ep_data_grp.create_group("obs")
            obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
            obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
            obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
            obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
            obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
            obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
            obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
            ep_data_grp.create_dataset("actions", data=actions)
            ep_data_grp.create_dataset("states", data=np.stack(states))
            ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
            ep_data_grp.create_dataset("rewards", data=rewards)
            ep_data_grp.create_dataset("dones", data=dones)

            num_success += 1

        num_replays += 1

        print(
            f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
        )
        print(f"  Total # no-op actions filtered out: {num_noops}")

    orig_data_file.close()
    new_data_file.close()
    print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=str,
        nargs="+",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        default=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite. Example: libero_spatial",
    )
    parser.add_argument(
        "--in-dir",
        type=pathlib.Path,
        help="Path to directory containing the unprocessed libero datasets. Example: ../LIBERO/libero/datasets",
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help="Path to directory to write the processed dataset. Example: ../LIBERO/libero/processed_datasets/",
        required=True,
    )
    args = parser.parse_args()

    for suite in args.suite:
        print(f"Regenerating {suite} dataset!")

        libero_target_dir: pathlib.Path = args.out_dir / suite

        # Create target directory
        if libero_target_dir.exists():
            user_input = input(
                f"Target directory already exists at path: {libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: "
            )
            if user_input != "y":
                exit()

        libero_target_dir.mkdir(parents=True, exist_ok=True)

        # Get task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite]()
        num_tasks_in_suite = task_suite.n_tasks

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=mp.cpu_count() // 8) as pool:
            for res in tqdm.tqdm(
                pool.imap_unordered(
                    partial(
                        process_one,
                        suite=suite,
                        libero_raw_data_dir=args.in_dir / suite,
                        libero_target_dir=libero_target_dir,
                    ),
                    range(num_tasks_in_suite),
                ),
                total=num_tasks_in_suite,
            ):
                print(res)

        print(f"Dataset regeneration complete! Saved new dataset at: {libero_target_dir}")


if __name__ == "__main__":
    main()
