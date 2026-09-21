"""Exercise actual policy inference methods with a cheap denoiser, not a simulator."""
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "policy")]
os.environ["HIPOLICY_BENCHMARK"] = sys.argv[1]
variant = sys.argv[2]
if variant == "dp":
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy as Policy
else:
    from diffusion_policy_3d.policy.dp3 import DP3 as Policy


class IdentityNormalizer:
    def normalize(self, value):
        return value.copy() if isinstance(value, dict) else value
    def unnormalize(self, value):
        return value
    def __getitem__(self, key):
        return self


class Denoiser:
    def __init__(self):
        self.batches = []
    def __call__(self, sample, timestep, **kw):
        self.batches.append(sample.shape[0])
        for key in ("global_cond_long", "global_cond_mid", "global_cond_short"):
            assert kw[key].shape[0] == sample.shape[0]
        return torch.arange(sample.shape[0], dtype=sample.dtype)[:, None, None].expand_as(sample).clone()


class Scheduler:
    def set_timesteps(self, steps):
        self.timesteps = range(steps)
    def step(self, model_output, timestep, sample, **kw):
        return SimpleNamespace(prev_sample=model_output)


policy = SimpleNamespace(normalizer=IdentityNormalizer(), obs_encoder=lambda x: x['agent_pos'][:, :2],
    parameters=lambda: iter([torch.zeros(1)]), device=torch.device('cpu'), dtype=torch.float32,
    action_dim=14, obs_feature_dim=2, n_obs_steps=7, n_action_steps=8, horizon=29,
    actual_horizon=8, n_obs_steps_use=3, short_interval=1, mid_interval=2, long_interval=3,
    obs_as_global_cond=True, no_hierarchical_cond=False, use_pc_color=False, condition_type='film',
    kwargs={}, model=Denoiser(), noise_scheduler=Scheduler(), num_inference_steps=2)
policy.conditional_sample = MethodType(Policy.conditional_sample, policy)
obs = {'agent_pos': torch.ones(2, 7, 14)}
if variant == 'dp3':
    obs['point_cloud'] = torch.ones(2, 7, 1024, 6)
for n in (1, 100):
    policy.model.batches.clear()
    result = Policy.predict_action(policy, obs, num_samples=n)
    assert policy.model.batches == [2*n, 2*n], policy.model.batches
    chunks = result['action_chunks']
    assert tuple(chunks.shape) == ((2, 24, 14) if n == 1 else (100, 2, 24, 14))
    if n == 100:
        assert chunks[0, 0, 0, 0] == 0 and chunks[0, 1, 0, 0] == 100
        assert chunks[99, 0, 0, 0] == 99 and chunks[99, 1, 0, 0] == 199
