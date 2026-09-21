import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ReleaseTests(unittest.TestCase):
    def test_python_syntax_and_no_private_paths(self):
        for p in ROOT.rglob('*'):
            if any(part in {'__pycache__', 'data', 'outputs', '.git', '.venv', 'build'} for part in p.relative_to(ROOT).parts):
                continue
            if p.suffix == '.py':
                ast.parse(p.read_text(), filename=str(p))
            if p.suffix in {'.py', '.sh', '.yml', '.yaml', '.md', '.json', '.toml', '.txt'}:
                self.assertIsNone(re.search(r'/(?:Users|home|DATA|mnt)/[^\s]+|https://[^\s"\']+\.openai\.azure\.com', p.read_text()), str(p))

    def test_computation_preserved(self):
        spec = importlib.util.spec_from_file_location('computation', ROOT / 'scripts/check_computation.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertGreater(module.check(), 20)

    def test_four_configurations(self):
        from execution.cli import parser, paths, training_config
        for benchmark in ('robotwin_v1', 'robotwin_v2'):
            for variant in ('dp', 'dp3'):
                args = parser().parse_args(['train', '--benchmark', benchmark, '--policy', variant, '--task', 'example'])
                cfg = training_config(args, paths(args))
                self.assertEqual((cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps), (29, 7, 8))
                self.assertEqual(list(cfg.shape_meta.action.shape), [14])
                self.assertEqual(cfg.dataloader.batch_size, 128)
                self.assertEqual(cfg.training.num_epochs, 600 if variant == 'dp' else 3000)
                self.assertEqual(cfg.optimizer.lr, 1e-4)
                self.assertEqual(list(cfg.optimizer.betas), [0.9, 0.999])
                self.assertEqual(cfg.policy.noise_scheduler._target_.split('.')[-1], 'DDPMScheduler')
                self.assertEqual(cfg.policy.num_inference_steps, 100)
                self.assertEqual(cfg.policy.noise_scheduler.prediction_type, 'epsilon' if variant == 'dp' else 'sample')
                self.assertEqual(cfg.policy.diffusion_step_embed_dim, 128 if variant == 'dp' else 64)
                self.assertEqual(cfg.logging.mode, 'offline')
                self.assertEqual(cfg.training.use_ema, not (benchmark == 'robotwin_v1' and variant == 'dp'))

    def test_any_working_directory(self):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/hipolicy.py'), 'train',
                                     '--benchmark', 'robotwin_v2', '--policy', 'dp3', '--task', 'example', '--dry-run'],
                                    cwd=cwd, capture_output=True, text=True, check=True)
        result = json.loads(result.stdout)
        self.assertEqual(Path(result['benchmark_root']), ROOT / 'benchmarks/robotwin_v2')
        self.assertIn('/robotwin_v2/dp3/', result['dataset'])
        self.assertFalse(result['entropy_guidance'])

    def test_v2_language_metadata_accepts_absolute_data_root(self):
        source = ROOT / 'benchmarks/robotwin_v2/description/utils/generate_episode_instructions.py'
        spec = importlib.util.spec_from_file_location('episode_language', source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example/demo_clean/scene_info.json'
            path.parent.mkdir(parents=True)
            path.write_text('{"episode0": {"info": {}}}')
            self.assertEqual(module.load_scene_info('example', 'demo_clean', directory), {'episode0': {'info': {}}})


if __name__ == '__main__':
    unittest.main()
