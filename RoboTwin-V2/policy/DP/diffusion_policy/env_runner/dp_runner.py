import torch
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DPRunner:

    def __init__(
        self,
        eval_episodes=20,
        max_steps=300,
        n_obs_steps=3,
        n_action_steps=8,
        fps=10,
        crf=22,
        tqdm_interval_sec=5.0,
        task_name=None,
    ):
        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.obs = deque(maxlen=n_obs_steps + 1)
        self.env = None
        self.entropy_list = []

    def stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def reset_obs(self):
        self.obs.clear()

    def reset_entropy_list(self):
        self.entropy_list = []

    def get_entropy_list(self):
        return self.entropy_list

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert len(self.obs) > 0, "no observation is recorded, please update obs first"

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs([obs[key] for obs in self.obs], self.n_obs_steps)

        return result

    def get_action(self, policy: BaseImagePolicy, observaton=None):
        device, dtype = policy.device, policy.dtype
        if observaton is not None:
            self.obs.append(observaton)  # update
        obs = self.get_n_steps_obs()

        # create obs dict
        np_obs_dict = dict(obs)
        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        # run policy
        with torch.no_grad():
            obs_dict_input = {}  # flush unused keys
            obs_dict_input["head_cam"] = obs_dict["head_cam"].unsqueeze(0)
            # obs_dict_input['front_cam'] = obs_dict['front_cam'].unsqueeze(0)
            obs_dict_input["left_cam"] = obs_dict["left_cam"].unsqueeze(0)
            obs_dict_input["right_cam"] = obs_dict["right_cam"].unsqueeze(0)
            obs_dict_input["agent_pos"] = obs_dict["agent_pos"].unsqueeze(0)

            # 并行采样 10 个样本，返回 shape: (num_samples=10, B=1, T=24, Da)
            action_dict = policy.predict_action(obs_dict_input, num_samples=100)["action"]
            # action_dict shape: (10, 1, 24, Da)
            #print("action_dict.shape:", action_dict.shape)
            action_dict_left = action_dict[:, :, :, :7]
            action_dict_right = action_dict[:, :, :, 7:14]

            # 计算方差和熵（在 num_samples 维度上）
            var = torch.var(action_dict, dim=0)  # (1, 24, Da)
            entropy = 0.5 * torch.log(2 * torch.pi * torch.e * var + 1e-6)
            entropy = entropy.mean()
            # print(f"action entropy_100: {entropy.item():.5f}")

            var_left = torch.var(action_dict_left, dim=0)  # (1, 24)
            entropy_left = 0.5 * torch.log(2 * torch.pi * torch.e * var_left + 1e-6)
            entropy_left = entropy_left.mean()
            # print(f"action entropy_left: {entropy_left.item():.5f}")

            var_right = torch.var(action_dict_right, dim=0)  # (1, 24)
            entropy_right = 0.5 * torch.log(2 * torch.pi * torch.e * var_right + 1e-6)
            entropy_right = entropy_right.mean()
            # print(f"action entropy_right: {entropy_right.item():.5f}")

            # var_10 = torch.var(action_dict_10, dim=0)  # (1, 24, Da)
            # entropy_10 = 0.5 * torch.log(2 * torch.pi * torch.e * var_10 + 1e-6)
            # entropy_10 = entropy_10.mean()
            # print(f"action entropy_10: {entropy_10.item():.5f}")

            entropy_max = max(entropy_left, entropy_right)

            self.entropy_list.append(entropy_max.item())

            # 计算均值作为最终 action
            action_mean = action_dict.mean(dim=0, keepdim=False)  # (1, 24, Da)
            action_dict = {"action": action_mean}

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
        action = np_action_dict["action"].squeeze(0)
        # print("action.shape:", action.shape)
        if entropy_right > -6.2 and entropy_right < -5.6:
            action_right = action[8:16, 7:14]
        elif entropy_right >= -5.6:
            action_right = action[:8, 7:14]
        else:
            action_right = action[16:24, 7:14]

        if entropy_left > -6.2 and entropy_left < -5.6:
            action_left = action[8:16, :7]
        elif entropy_left >= -5.6:
            action_left = action[:8, :7]
        else:
            action_left = action[16:24, :7]
        action = action[16:24, :]
        return action
