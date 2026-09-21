import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from execution.entropy import expand_batch, unpack_candidates, predict, scale_for_entropy, select_candidates


class ExecutionTests(unittest.TestCase):
    def test_threshold_boundaries(self):
        self.assertEqual(scale_for_entropy(-6.00001)[0], "short")
        self.assertEqual(scale_for_entropy(-6.0)[0], "mid")
        self.assertEqual(scale_for_entropy(-5.5)[0], "long")
        with self.assertRaises(ValueError):
            scale_for_entropy(float("nan"))

    def test_first_candidate_global_scale_and_all_dimensions(self):
        # Three observations require three different scales in the same call.
        z = torch.arange(100, dtype=torch.float64)
        z = (z - z.mean()) / z.std()
        variances = torch.tensor([(math.exp(2*h)-1e-6)/(2*math.pi*math.e) for h in (-6.3, -5.8, -5.0)])
        tags = torch.arange(24, dtype=torch.float64)[None, None, :, None].expand(100, 3, 24, 14)
        candidates = tags + z[:, None, None, None] * variances.sqrt()[None, :, None, None]
        action, entropy = select_candidates(candidates, True)
        torch.testing.assert_close(entropy, torch.tensor([-6.3, -5.8, -5.0], dtype=torch.float64))
        for b, start in enumerate((16, 8, 0)):
            torch.testing.assert_close(action[b], candidates[0, b, start:start+8])
            self.assertFalse(torch.equal(action[b], candidates[:, b, start:start+8].mean(0)))

    def test_zero_variance_and_off(self):
        candidates = torch.zeros(100, 2, 24, 14)
        selected, entropy = select_candidates(candidates, True)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertEqual(tuple(selected.shape), (2, 8, 14))
        single = torch.arange(24*14.).reshape(1, 1, 24, 14)
        selected, entropy = select_candidates(single, False)
        self.assertIsNone(entropy)
        torch.testing.assert_close(selected, single[0, :, 16:24])

    def test_parallel_batch_order(self):
        observations = torch.tensor([[10., 11.], [20., 21.]])
        expanded = expand_batch(observations, 100)
        self.assertEqual(tuple(expanded.shape), (200, 2))
        torch.testing.assert_close(expanded[:100], observations[0].expand(100, -1))
        restored = unpack_candidates(expanded, 2, 100)
        torch.testing.assert_close(restored[:, 0], observations[0].expand(100, -1))
        torch.testing.assert_close(restored[:, 1], observations[1].expand(100, -1))

    def test_one_policy_call_per_decision(self):
        class FakePolicy:
            def __init__(self):
                self.calls = []
            def predict_action(self, obs, num_samples):
                self.calls.append(num_samples)
                shape = (1, 24, 14) if num_samples == 1 else (num_samples, 1, 24, 14)
                return {"action_chunks": torch.zeros(shape)}
        policy = FakePolicy()
        predict(policy, {}, False)
        predict(policy, {}, True)
        self.assertEqual(policy.calls, [1, 100])

    def test_real_policy_sampling_interfaces(self):
        # Import each original namespace/profile in a separate process.
        for benchmark in ("robotwin_v1", "robotwin_v2"):
            for variant in ("dp", "dp3"):
                with self.subTest(benchmark=benchmark, variant=variant):
                    subprocess.run([sys.executable, str(ROOT / "tests/sampling_probe.py"), benchmark, variant], check=True)

    def test_checkpoint_compatibility(self):
        for benchmark in ("robotwin_v1", "robotwin_v2"):
            for variant in ("dp", "dp3"):
                with self.subTest(benchmark=benchmark, variant=variant):
                    subprocess.run([sys.executable, str(ROOT / "tests/checkpoint_probe.py"), benchmark, variant], check=True)


if __name__ == "__main__":
    unittest.main()
