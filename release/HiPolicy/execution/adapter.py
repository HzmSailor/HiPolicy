"""Shared inference adapter; simulator-specific observation cadence stays intact."""
from collections import deque
import numpy as np

from execution.checkpoint import load_policy
from execution.entropy import predict


class PolicyAdapter:
    def __init__(self, checkpoint, policy_name, entropy_guidance=False, device="cuda:0"):
        self.policy, self.cfg = load_policy(checkpoint, device)
        self.policy_name = policy_name
        self.entropy_guidance = entropy_guidance
        self.n_obs_steps = self.policy.n_obs_steps
        self.obs = deque(maxlen=self.n_obs_steps + 1)
        self.runner = self  # Original V1 evaluation resets dp.runner.reset_obs().
        self.env_runner = self  # Original V1 DP3 evaluation uses this name.
        self.last_entropy = None

    def update_obs(self, observation):
        self.obs.append(observation)

    def reset_obs(self):
        self.obs.clear()
        self.last_entropy = None

    def get_last_obs(self):
        return self.obs[-1]

    def get_action(self, observation=None):
        import torch
        if observation is not None:
            self.update_obs(observation)
        if not self.obs:
            raise ValueError("Record an observation before inference")
        keys = self.cfg.shape_meta.obs.keys()
        inputs = {}
        for key in keys:
            values = [obs[key] for obs in self.obs][-self.n_obs_steps:]
            padded = [values[0]] * (self.n_obs_steps - len(values)) + values
            inputs[key] = torch.from_numpy(np.stack(padded)).to(self.policy.device).unsqueeze(0)
        with torch.no_grad():
            actions, entropy = predict(self.policy, inputs, self.entropy_guidance)
        self.last_entropy = None if entropy is None else entropy.detach().cpu().tolist()
        return actions[0].detach().cpu().numpy()


def get_model(args):
    if args["left_arm_dim"] + args["right_arm_dim"] + 2 != 14:
        raise ValueError("The released HiPolicy routes require 14D ALOHA actions")
    return PolicyAdapter(args["checkpoint"], args["variant"], args["entropy_guidance"])


def encode_obs(observation, variant):
    obs = {"agent_pos": observation["joint_action"]["vector"]}
    if variant == "dp":
        for key, camera in (("head_cam", "head_camera"), ("left_cam", "left_camera"), ("right_cam", "right_camera")):
            obs[key] = np.moveaxis(observation["observation"][camera]["rgb"], -1, 0) / 255
    else:
        obs["point_cloud"] = observation["pointcloud"]
    return obs


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation, model.policy_name)
    instruction = TASK_ENV.get_instruction()
    actions = model.get_action(obs)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        model.update_obs(encode_obs(observation, model.policy_name))


def reset_model(model):
    model.reset_obs()
