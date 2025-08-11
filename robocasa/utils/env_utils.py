from functools import partial

import gymnasium as gym
import imageio
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from termcolor import colored
from tqdm import tqdm

from robocasa.utils.dataset_registry import (
    MULTI_STAGE_TASK_DATASETS,
    SINGLE_STAGE_TASK_DATASETS,
)


def create_env(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    seed=None,
    render_onscreen=False,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=(not render_onscreen),
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env


def create_robocasa_gym_env(
    env_name,
    seed=None,
    # robosuite-related configs
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    render_onscreen=False,
    # robocasa-related configs
    obj_instance_split="B",
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        ignore_done=False,
        use_object_obs=True,
        use_camera_obs=(not render_onscreen),
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    env = GymWrapper(
        env,
        flatten_obs=False,
        keys=[
            "robot0_base_pos",
            "robot0_base_quat",
            "robot0_eef_pos",
            "robot0_base_to_eef_pos",
            "robot0_eef_quat",
            "robot0_base_to_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_agentview_left_image",
            "robot0_agentview_right_image",
            "robot0_eye_in_hand_image",
        ],
    )
    env = RoboCasaWrapper(env)
    return env


def load_robocasa_gym_env(env_name, n_envs=1, **kwargs):
    env_fns = [partial(create_robocasa_gym_env, env_name=env_name, **kwargs) for _ in range(n_envs)]
    if n_envs == 1:
        return gym.vector.SyncVectorEnv(env_fns)
    else:
        return gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )


def run_random_rollouts(env, num_rollouts, num_steps, video_path=None):
    video_writer = None
    if video_path is not None:
        video_writer = imageio.get_writer(video_path, fps=20)

    info = {}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(num_rollouts)):
        obs = env.reset()
        for step_i in range(num_steps):
            # sample and execute random action
            action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
            obs, _, _, _ = env.step(action)

            if video_writer is not None:
                video_img = env.sim.render(height=512, width=768, camera_name="robot0_agentview_center")[::-1]
                video_writer.append_data(video_img)

            if env._check_success():
                num_success_rollouts += 1
                break

    if video_writer is not None:
        video_writer.close()
        print(colored(f"Saved video of rollouts to {video_path}", color="yellow"))

    info["num_success_rollouts"] = num_success_rollouts

    return info


class RoboCasaWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def language_instruction(self):
        return self.env.get_ep_meta()["lang"]

    def reset(self, seed=None, options=None):
        obs, _ = super().reset(seed=seed, options=options)
        info = {}
        info["success"] = self.is_success()["task"]
        return obs, info

    def render(self, mode="rgb_array"):
        return self.env.unwrapped.sim.render(camera_name="robot0_agentview_center", height=512, width=512)[::-1]

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return {"task": succ}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["success"] = self.is_success()["task"]
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()


if __name__ == "__main__":
    # select random task to run rollouts for
    env_name = np.random.choice(list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS))
    env = create_eval_env(env_name=env_name)
    info = run_random_rollouts(env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4")
