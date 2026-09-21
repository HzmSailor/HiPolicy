"""Synthetic legacy-format checkpoints; compact models keep CPU checks practical."""
import copy
import importlib
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'policy')]
benchmark, variant = sys.argv[1:]
os.environ['HIPOLICY_BENCHMARK'] = benchmark
from execution.cli import parser, paths, training_config
from execution.checkpoint import load_policy, load_training_payload
import dill
import hydra
import torch
from omegaconf import OmegaConf

torch.set_num_threads(1)
args = parser().parse_args(['train', '--benchmark', benchmark, '--policy', variant, '--task', 'fixture'])
cfg = training_config(args, paths(args))
cfg.policy.down_dims = [32, 64]
cfg.policy.diffusion_step_embed_dim = 32
if variant == 'dp':
    cfg.policy.shape_meta.obs.head_cam.shape = [3, 16, 16]
    cfg.policy.obs_encoder.shape_meta = cfg.policy.shape_meta
    package = 'diffusion_policy'
else:
    # Exercise loading original DP3 DDIM checkpoints into approved DDPM inference.
    cfg.policy.noise_scheduler._target_ = 'diffusers.schedulers.scheduling_ddim.DDIMScheduler'
    cfg.policy.num_inference_steps = 10
    package = 'diffusion_policy_3d'
cfg.shape_meta = cfg.policy.shape_meta
cfg.training.use_ema = True
model = hydra.utils.instantiate(cfg.policy)
single = importlib.import_module(package + '.model.common.normalizer').SingleFieldLinearNormalizer
for key in ['action', *cfg.shape_meta.obs.keys()]:
    model.normalizer[key] = single.create_identity()
state = model.state_dict()
weight_key = next(k for k, v in state.items() if v.numel() and k.startswith('model.'))
ema_state = {('module._orig_mod.' + k): v.clone() for k, v in state.items()}
ema_state['module._orig_mod.' + weight_key].fill_(0.123)
payload = {'cfg': cfg, 'state_dicts': {'model': state, 'ema_model': ema_state}, 'pickles': {}}
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'legacy.ckpt'
    torch.save(payload, path, pickle_module=dill)
    loaded, saved_cfg = load_policy(path, 'cpu')
    assert loaded.noise_scheduler.__class__.__name__ == 'DDPMScheduler'
    assert loaded.num_inference_steps == 100
    assert loaded.noise_scheduler.config.prediction_type == ('epsilon' if variant == 'dp' else 'sample')
    torch.testing.assert_close(loaded.state_dict()[weight_key], torch.full_like(state[weight_key], 0.123))
    assert set(loaded.state_dict()) == set(state)
    # Missing real neural weights must fail rather than being silently ignored.
    del ema_state['module._orig_mod.' + weight_key]
    torch.save(payload, path, pickle_module=dill)
    try:
        load_policy(path, 'cpu')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Missing neural parameters were accepted')

    # Exercise actual trainer checkpoint I/O with small stand-in state holders.
    # This checks relocation/resumption, not the training algorithm or GPU setup.
    module = importlib.import_module(f'hipolicy_compat.{benchmark}.{variant}_workspace')
    cls = module.RobotWorkspace if variant == 'dp' else module.TrainDP3Workspace
    workspace = cls.__new__(cls)
    workspace._output_dir = str(Path(directory) / 'relocated')
    workspace.global_step, workspace.epoch = 0, 0
    workspace.model = torch.nn.Linear(2, 2)
    workspace.optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=0.002)
    workspace.model(torch.ones(1, 2)).sum().backward()
    workspace.optimizer.step()
    saved_model = copy.deepcopy(workspace.model.state_dict())
    saved_optimizer = copy.deepcopy(workspace.optimizer.state_dict())
    payload = {'cfg': cfg, 'state_dicts': {'model': saved_model, 'optimizer': saved_optimizer},
               'pickles': {'_output_dir': dill.dumps(str(Path(directory) / 'previous')),
                           'global_step': dill.dumps(17), 'epoch': dill.dumps(3)}}
    torch.save(payload, path, pickle_module=dill)
    workspace.model = torch.nn.Linear(2, 2)
    workspace.optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=0.5)
    workspace.load_checkpoint(path, map_location='cpu')
    assert workspace.output_dir == str(Path(directory) / 'relocated')
    assert (workspace.global_step, workspace.epoch) == (17, 3)
    for key, value in saved_model.items():
        torch.testing.assert_close(workspace.model.state_dict()[key], value, rtol=0, atol=0)
    assert workspace.optimizer.param_groups[0]['lr'] == 0.002
    for parameter, state in saved_optimizer['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(workspace.optimizer.state_dict()['state'][parameter][key], value, rtol=0, atol=0)
    assert load_training_payload(path, cfg)['pickles']['epoch'] == payload['pickles']['epoch']
    incompatible = copy.deepcopy(cfg)
    incompatible.policy.diffusion_step_embed_dim += 1
    try:
        load_training_payload(path, incompatible)
    except ValueError:
        pass
    else:
        raise AssertionError('Incompatible training configuration was accepted')
