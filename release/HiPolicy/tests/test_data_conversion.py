"""Run all four original converters on small, distinguishable temporal fixtures."""
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ConversionTests(unittest.TestCase):
    def test_four_routes_preserve_temporal_alignment(self):
        from execution.cli import convert, parser

        vectors = np.arange(5*14, dtype=np.float32).reshape(5, 14)
        points = np.arange(5*1024*6, dtype=np.float32).reshape(5, 1024, 6)
        images = np.zeros((5, 4, 6, 3), dtype=np.uint8)
        images[..., 0], images[..., 1], images[..., 2] = 11, 22, 33
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1 = root / 'v1/episode0'; v1.mkdir(parents=True)
            for i in range(5):
                with (v1 / f'{i}.pkl').open('wb') as f:
                    pickle.dump({'joint_action': vectors[i], 'endpose': vectors[i], 'pointcloud': points[i],
                                 'observation': {'head_camera': {'rgb': images[i]}}}, f)
            v2 = root / 'v2/data'; v2.mkdir(parents=True)
            with h5py.File(v2 / 'episode0.hdf5', 'w') as f:
                for key, values in [('vector', vectors), ('left_gripper', vectors[:, 6]),
                                    ('right_gripper', vectors[:, 13]), ('left_arm', vectors[:, :6]), ('right_arm', vectors[:, 7:13])]:
                    f.create_dataset('joint_action/' + key, data=values)
                f.create_dataset('pointcloud', data=points)
                encoded = [cv2.imencode('.png', image)[1].tobytes() for image in images]
                f.create_dataset('observation/head_camera/rgb', data=np.array(encoded, dtype=f'S{max(map(len, encoded))}'))
            for benchmark in ('robotwin_v1', 'robotwin_v2'):
                for variant in ('dp', 'dp3'):
                    with self.subTest(benchmark=benchmark, variant=variant):
                        source = ROOT / 'benchmarks' / benchmark / 'script' / (f'pkl2zarr_{variant}.py' if benchmark.endswith('v1') else f'process_{variant}.py')
                        spec = importlib.util.spec_from_file_location('converter', source)
                        converter = importlib.util.module_from_spec(spec); spec.loader.exec_module(converter)
                        destination = root / f'{benchmark}_{variant}.zarr'
                        environment = {'HIPOLICY_CONVERT_INPUT': str(root / ('v1' if benchmark.endswith('v1') else 'v2')),
                                       'HIPOLICY_CONVERT_OUTPUT': str(destination)}
                        args = parser().parse_args(['convert', '--benchmark', benchmark, '--policy', variant,
                                                   '--task', 'fixture', '--episodes', '1'])
                        paths = {'benchmark_root': source.parents[1], 'dataset': destination,
                                 'raw_dataset': Path(environment['HIPOLICY_CONVERT_INPUT'])}
                        with patch.dict(os.environ, environment), redirect_stdout(io.StringIO()):
                            convert(args, paths)
                            self.assertEqual(os.environ['HIPOLICY_CONVERT_OUTPUT'], str(destination))
                        store = zarr.open(str(destination), 'r')
                        v1_mode = benchmark.endswith('v1')
                        np.testing.assert_array_equal(store['data/state'][:], vectors if v1_mode else vectors[:-1])
                        np.testing.assert_array_equal(store['data/action'][:], vectors if v1_mode else vectors[1:])
                        np.testing.assert_array_equal(store['meta/episode_ends'][:], [5 if v1_mode else 4])
                        if variant == 'dp3':
                            np.testing.assert_array_equal(store['data/point_cloud'][:], points if v1_mode else points[:-1])
                        else:
                            expected = images if v1_mode else images[:-1]
                            np.testing.assert_array_equal(store['data/head_camera'][:], np.moveaxis(expected, -1, 1))
                        with patch.dict(os.environ, environment), patch.object(sys, 'argv', ['converter', 'fixture', 'D435', '1']):
                            with self.assertRaises(FileExistsError):
                                converter.main()
                        with self.assertRaises(FileExistsError):
                            convert(args, paths)

    def test_failed_conversion_can_be_retried_without_partial_output(self):
        from execution.data import conversion_output

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / 'processed/fixture.zarr'
            raw = root / 'raw.pkl'
            raw.write_bytes(b'keep raw demonstrations')
            with patch.dict(os.environ, {'HIPOLICY_CONVERT_OUTPUT': 'previous'}):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    with conversion_output(destination, 'dp', 1) as staged:
                        staged.mkdir()
                        (staged / 'partial').write_bytes(b'partial output')
                        raise RuntimeError('interrupted')
                self.assertEqual(os.environ['HIPOLICY_CONVERT_OUTPUT'], 'previous')
                self.assertFalse(destination.exists())
                self.assertEqual(list(destination.parent.iterdir()), [])
                with conversion_output(destination, 'dp', 1) as staged:
                    store = zarr.open(str(staged), mode='w')
                    store.create_dataset('data/state', data=np.zeros((2, 14), dtype=np.float32))
                    store.create_dataset('data/action', data=np.ones((2, 14), dtype=np.float32))
                    store.create_dataset('data/head_camera', data=np.zeros((2, 3, 4, 6), dtype=np.uint8))
                    store.create_dataset('meta/episode_ends', data=np.array([2], dtype=np.int64))
                self.assertEqual(list(destination.parent.iterdir()), [destination])
                self.assertEqual(raw.read_bytes(), b'keep raw demonstrations')
                np.testing.assert_array_equal(zarr.open(str(destination), mode='r')['data/action'][:], np.ones((2, 14)))

    def test_incomplete_store_is_not_published(self):
        from execution.data import conversion_output

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'fixture.zarr'
            with self.assertRaisesRegex(ValueError, 'episode boundaries'):
                with conversion_output(destination, 'dp3', 2) as staged:
                    zarr.open(str(staged), mode='w').create_dataset('meta/episode_ends', data=np.array([4]))
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
