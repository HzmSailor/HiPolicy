#!/usr/bin/env python3
"""Strictly load a trusted original checkpoint and smoke-test both EG modes."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'policy')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=('robotwin_v1', 'robotwin_v2'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    os.environ['HIPOLICY_BENCHMARK'] = args.benchmark
    from execution.checkpoint import load_policy
    from execution.entropy import predict
    import torch
    policy, cfg = load_policy(str(args.checkpoint), args.device)
    observations = {k: torch.zeros((1, cfg.n_obs_steps, *meta.shape), device=args.device)
                    for k, meta in cfg.shape_meta.obs.items()}
    before = {k: tuple(v.shape) for k, v in policy.state_dict().items()}
    for enabled in (False, True):
        with torch.no_grad():
            actions, entropy = predict(policy, observations, enabled)
        assert actions.shape == (1, 8, 14) and torch.isfinite(actions).all()
        assert before == {k: tuple(v.shape) for k, v in policy.state_dict().items()}
        print(f'EG={enabled}: action={tuple(actions.shape)}, entropy={entropy}')
    print('Strict checkpoint load and synthetic inference passed; this is not a task-success evaluation.')


if __name__ == '__main__':
    main()
