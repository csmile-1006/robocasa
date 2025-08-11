import gc
import os
import time

import numpy as np
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from robocasa.lifelong.utils import *
from robocasa.utils.env_utils import load_robocasa_gym_env
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS


def get_env_horizon(env_name):
    if env_name in SINGLE_STAGE_TASK_DATASETS:
        ds_config = SINGLE_STAGE_TASK_DATASETS[env_name]
    elif env_name in MULTI_STAGE_TASK_DATASETS:
        ds_config = MULTI_STAGE_TASK_DATASETS[env_name]
    else:
        raise ValueError(f"Environment {env_name} not found in dataset registry")
    return ds_config["horizon"]


def raw_obs_to_tensor_obs(obs, task_emb, cfg):
    """
    Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)

    data = {
        "obs": {},
        "task_emb": task_emb.repeat(env_num, 1),
    }

    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    for k in range(env_num):
        for obs_name in all_obs_keys:
            data["obs"][obs_name].append(
                ObsUtils.process_obs(
                    torch.from_numpy(obs[k][cfg.data.obs_key_mapping[obs_name]]),
                    obs_key=obs_name,
                ).float()
            )

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    return data


def evaluate_one_task_success(cfg, algo, task_name):
    """
    Evaluate a single task's success rate
    sim_states: if not None, will keep track of all simulated states during
                evaluation, mainly for visualization and debugging purpose
    task_str:   the key to access sim_states dictionary
    """
    with Timer() as t:
        if cfg.lifelong.algo == "PackNet":  # need preprocess weights for PackNet
            algo = algo.get_eval_algo(task_name)

        algo.eval()

        # initiate evaluation envs
        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # Try to handle the frame buffer issue
        env_creation = False

        count = 0
        while not env_creation and count < 5:
            try:
                env = load_robocasa_gym_env(env_name=task_name, n_envs=env_num)
                # if env_num == 1:
                #     env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
                # else:
                #     env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
                env_creation = True
            except Exception as e:
                time.sleep(5)
                count += 1
        if count >= 5:
            raise RuntimeError("Failed to create environment")

        ### Evaluation loop
        # get fixed init states to control the experiment randomness
        # init_states_path = os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file)
        # init_states = torch.load(init_states_path)
        num_success = 0
        max_steps = get_env_horizon(task_name)
        for i in range(eval_loop_num):
            env.reset()
            task_emb = env.language_instruction
            # indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
            # init_states_ = init_states[indices]

            dones = [False] * env_num
            steps = 0
            algo.reset()
            # obs = env.set_init_state(init_states_)

            # dummy actions [env_num, 7] all zeros for initial physics simulation
            # dummy = np.zeros((env_num, 7))
            # for _ in range(5):
            #     obs, _, _, _ = env.step(dummy)

            while steps < max_steps:
                steps += 1

                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = algo.policy.get_action(data)

                obs, reward, done, info = env.step(actions)

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or info["success"][k]

                if all(dones):
                    break

            # a new form of success record
            for k in range(env_num):
                if i * env_num + k < cfg.eval.n_eval:
                    num_success += int(dones[k])

        success_rate = num_success / cfg.eval.n_eval
        env.close()
        gc.collect()
    print(f"[info] evaluate task {task_name} takes {t.get_elapsed_time():.1f} seconds")
    return success_rate


def evaluate_success(cfg, algo, task_names, result_summary=None):
    """
    Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for task_name in task_names:
        success_rate = evaluate_one_task_success(cfg, algo, task_name)
        successes.append(success_rate)
    return np.array(successes)


def evaluate_multitask_training_success(cfg, algo, benchmark, task_ids):
    """
    Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        success_rate = evaluate_one_task_success(cfg, algo, task_i, task_emb, i)
        successes.append(success_rate)
    return np.array(successes)


@torch.no_grad()
def evaluate_loss(cfg, algo, datasets):
    """
    Evaluate the loss on all datasets.
    """
    algo.eval()
    losses = []
    for i, dataset in tqdm(enumerate(datasets), desc="Evaluating loss", leave=False, total=len(datasets)):
        if cfg.lifelong.algo == "PackNet":  # need preprocess weights for PackNet
            algo = algo.get_eval_algo(task_id=i)

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.num_workers,
            shuffle=False,
        )
        test_loss = 0
        for idx, data in tqdm(enumerate(dataloader), desc=f"Evaluating loss {i}", leave=False, total=len(dataloader)):
            data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
            loss = algo.policy.compute_loss(data)
            test_loss += loss.item()
            if idx >= 10:
                break
        test_loss /= len(dataloader)
        losses.append(test_loss)
    return np.array(losses)
