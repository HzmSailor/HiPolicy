#!/usr/bin/env python3
"""Compare original and release policies in isolated processes.

structure: construct full-size models on the meta device and compare all keys,
shapes, buffers and module types. numerical: compare fixed-input inference,
loss, gradients and one AdamW update with the ORIGINAL scheduler/settings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-root', required=True, type=Path)
    p.add_argument('--benchmark', required=True, choices=('robotwin_v1', 'robotwin_v2'))
    p.add_argument('--policy', required=True, choices=('dp', 'dp3'))
    p.add_argument('--mode', choices=('structure', 'numerical'), default='structure')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--worker', choices=('original', 'release'), help=argparse.SUPPRESS)
    return p.parse_args()


def check_import_sources(package, expected_root):
    """Fail if either worker accidentally imports the other policy tree."""
    expected_root = expected_root.resolve()
    for name, module in list(sys.modules.items()):
        if name == package or name.startswith(package + '.'):
            filename = getattr(module, '__file__', None)
            if filename and expected_root not in Path(filename).resolve().parents:
                raise AssertionError(f'Policy import escaped the selected source tree: {name}')


def worker(args):
    sys.path[:0] = [str(ROOT), str(ROOT / 'policy')]
    os.environ['HIPOLICY_BENCHMARK'] = args.benchmark
    from execution.cli import parser, paths, training_config
    cli_args = parser().parse_args(['train', '--benchmark', args.benchmark, '--policy', args.policy, '--task', 'block_hammer_beat'])
    cfg = training_config(cli_args, paths(cli_args))
    version = 'RoboTwin-V1' if args.benchmark.endswith('v1') else 'RoboTwin-V2'
    if args.policy == 'dp':
        relative = 'policy/Diffusion-Policy' if version.endswith('V1') else 'policy/DP'
        package = 'diffusion_policy'
        original_config = 'robot_dp.yaml' if version.endswith('V1') else 'robot_dp_14.yaml'
    else:
        relative = 'policy/3D-Diffusion-Policy/3D-Diffusion-Policy' if version.endswith('V1') else 'policy/DP3/3D-Diffusion-Policy'
        package, original_config = 'diffusion_policy_3d', 'robot_dp3.yaml'
    source = args.original_root / version / relative
    import yaml
    from omegaconf import OmegaConf
    original_cfg = yaml.safe_load((source / package / 'config' / original_config).read_text())
    # Separate packaging parity from the explicitly approved parameter changes.
    cfg.policy.noise_scheduler = OmegaConf.create(original_cfg['policy']['noise_scheduler'])
    cfg.policy.num_inference_steps = original_cfg['policy']['num_inference_steps']
    cfg.optimizer = OmegaConf.create(original_cfg['optimizer'])
    if args.worker == 'original':
        sys.path.insert(0, str(source))
    import hydra
    import torch
    torch.manual_seed(31415)
    torch.set_num_threads(1)
    if args.mode == 'structure':
        # torchvision Normalize inspects real std values and cannot run on meta.
        # Construct the small RGB encoder on CPU; keep the large U-Net on meta.
        encoder = hydra.utils.instantiate(cfg.policy.obs_encoder) if args.policy == 'dp' else None
        with torch.device('meta'):
            model = hydra.utils.instantiate(cfg.policy, **({'obs_encoder': encoder} if encoder is not None else {}))
        check_import_sources(package, source if args.worker == 'original' else ROOT / 'policy')
        payload = {
            'state': {k: list(v.shape) for k, v in model.state_dict().items()},
            'buffers': {k: list(v.shape) for k, v in model.named_buffers()},
            'modules': {k: type(v).__module__ + '.' + type(v).__name__ for k, v in model.named_modules()},
        }
        args.output.write_text(json.dumps(payload, indent=2) + '\n')
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Numerical GPU comparison requires CUDA; use --mode structure for local checks')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = hydra.utils.instantiate(cfg.policy).to(args.device)
    check_import_sources(package, source if args.worker == 'original' else ROOT / 'policy')
    import importlib
    single = importlib.import_module(package + '.model.common.normalizer').SingleFieldLinearNormalizer
    for key in ['action', *cfg.shape_meta.obs.keys()]:
        model.normalizer[key] = single.create_identity()
    model.to(args.device)
    torch.manual_seed(1234)
    observations = {k: torch.rand((2, 7, *meta.shape), device=args.device) for k, meta in cfg.shape_meta.obs.items()}
    batch = {'obs': observations, 'action': torch.randn(2, 29, 14, device=args.device)}
    model.eval()
    with torch.no_grad():
        torch.manual_seed(5678)
        action = model.predict_action(observations)['action'].detach().cpu()
    model.train()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    torch.manual_seed(9012)
    losses = model.compute_loss(batch)
    loss = losses[0] if isinstance(losses, tuple) else losses
    loss.backward()
    def hashes(items):
        return {k: hashlib.sha256(v.detach().contiguous().cpu().numpy().tobytes()).hexdigest() for k, v in items if v is not None}
    gradients = hashes((k, v.grad) for k, v in model.named_parameters())
    optimizer.step()
    payload = {'action': action, 'loss': loss.detach().cpu(), 'gradients': gradients,
               'updated_state': hashes(model.state_dict().items())}
    torch.save(payload, args.output)


def main():
    args = arguments()
    args.original_root = args.original_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.worker:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    extension = '.json' if args.mode == 'structure' else '.pt'
    files = []
    for source in ('original', 'release'):
        destination = args.output / (source + extension)
        command = [sys.executable, str(Path(__file__).resolve()), '--original-root', str(args.original_root),
                   '--benchmark', args.benchmark, '--policy', args.policy, '--mode', args.mode, '--device', args.device,
                   '--output', str(destination), '--worker', source]
        subprocess.run(command, check=True)
        files.append(destination)
    if args.mode == 'structure':
        a, b = [json.loads(p.read_text()) for p in files]
        assert a == b, 'Parameter/buffer/module structure differs'
    else:
        import torch
        a, b = [torch.load(p, map_location='cpu', weights_only=True) for p in files]
        torch.testing.assert_close(a['action'], b['action'], rtol=0, atol=0)
        torch.testing.assert_close(a['loss'], b['loss'], rtol=0, atol=0)
        assert a['gradients'] == b['gradients'], 'Gradient hashes differ'
        assert a['updated_state'] == b['updated_state'], 'One-step parameter update differs'
    result = {'benchmark': args.benchmark, 'policy': args.policy, 'mode': args.mode,
              'device': args.device if args.mode == 'numerical' else 'meta (RGB encoder on CPU)',
              'baseline': 'original scheduler and optimizer', 'passed': True}
    if args.mode == 'numerical':
        result.update(action_shape=list(a['action'].shape),
                      max_action_abs_difference=float((a['action'] - b['action']).abs().max()),
                      loss_abs_difference=float((a['loss'] - b['loss']).abs()),
                      gradient_tensor_count=len(a['gradients']),
                      updated_state_tensor_count=len(a['updated_state']))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
