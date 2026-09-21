from typing import Dict
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb
import time

class RobotImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            batch_size=128,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            keys=['head_camera', 'state', 'action']
        )
            
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        
        # 优化内存分配 - 使用更大的预分配buffer提升性能
        effective_batch_size = max(batch_size, 128)  # 增加预分配大小以提高效率
        self.buffers = {
            k: np.zeros((effective_batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        
        # 创建pinned memory缓冲区提升数据传输效率
        self.buffers_torch = {}
        for k, v in self.buffers.items():
            tensor = torch.from_numpy(v)
            # 优化：先转换为float32再 pin_memory
            if k == 'head_camera' and v.dtype == np.uint8:
                tensor = tensor.float().div_(255.0)  # 预先归一化
            tensor = tensor.pin_memory()
            self.buffers_torch[k] = tensor
            
        # GPU缓冲区管理
        self._gpu_buffers = {}
        self._device_cache = None
        self._current_stream = None
        
        # 性能监控
        self._perf_stats = {
            'total_postprocess_time': 0.0,
            'transfer_time': 0.0,
            'preprocess_time': 0.0,
            'batch_count': 0
        }

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['head_cam'] = get_image_range_normalizer()
        normalizer['front_cam'] = get_image_range_normalizer()
        normalizer['left_cam'] = get_image_range_normalizer()
        normalizer['right_cam'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32) # (agent_posx2, block_posex3)
        head_cam = np.moveaxis(sample['head_camera'],-1,1)/255
        # front_cam = np.moveaxis(sample['front_camera'],-1,1)/255
        # left_cam = np.moveaxis(sample['left_camera'],-1,1)/255
        # right_cam = np.moveaxis(sample['right_camera'],-1,1)/255

        data = {
            'obs': {
                'head_cam': head_cam, # T, 3, H, W
                # 'front_cam': front_cam, # T, 3, H, W
                # 'left_cam': left_cam, # T, 3, H, W
                # 'right_cam': right_cam, # T, 3, H, W
                'agent_pos': agent_pos, # T, D
            },
            'action': sample['action'].astype(np.float32) # T, D
        }
        return data
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            actual_batch_size = len(idx)
            # 动态分配buffer以匹配实际batch size
            if actual_batch_size != self.batch_size:
                temp_buffers = {}
                for k, v in self.sampler.replay_buffer.items():
                    temp_buffers[k] = np.zeros((actual_batch_size, self.sampler.sequence_length, *v.shape[1:]), dtype=v.dtype)
                    batch_sample_sequence(temp_buffers[k], v, self.sampler.indices, idx, self.sampler.sequence_length)
                
                # 转换为torch tensor并返回
                result = {}
                for k, v in temp_buffers.items():
                    tensor = torch.from_numpy(v)
                    if k == 'head_camera' and v.dtype == np.uint8:
                        tensor = tensor.float().div_(255.0)
                    result[k] = tensor
                return result
            else:
                # 使用预分配的buffer
                for k, v in self.sampler.replay_buffer.items():
                    batch_sample_sequence(self.buffers[k][:actual_batch_size], v, self.sampler.indices, idx, self.sampler.sequence_length)
                
                # 返回对应大小的slice
                result = {}
                for k, v in self.buffers_torch.items():
                    result[k] = v[:actual_batch_size]
                return result
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        # 高效数据传输和预处理优化版本，添加性能监控
        start_time = time.time()
        self._perf_stats['batch_count'] += 1
        
        if self._current_stream is None:
            self._current_stream = torch.cuda.Stream(device)
            
        # 缓存GPU缓冲区以减少重复分配
        if device != self._device_cache:
            self._gpu_buffers.clear()
            self._device_cache = device
            
        if 'obs' in samples and 'action' in samples:
            # 已经是处理过的格式，使用异步传输优化
            obs = samples['obs']
            if isinstance(obs, dict):
                processed_obs = {}
                # 使用专用CUDA流实现异步传输
                with torch.cuda.stream(self._current_stream):
                    for key, value in obs.items():
                        if hasattr(value, 'to'):
                            # 优化内存格式和类型转换
                            if key == 'head_cam' and value.dtype == torch.uint8:
                                # 安全的channels_last转换，仅对4D张量使用
                                if value.ndim == 4:
                                    processed_obs[key] = value.to(device, dtype=torch.float32, 
                                                                 memory_format=torch.channels_last, 
                                                                 non_blocking=True).div_(255.0)
                                else:
                                    processed_obs[key] = value.to(device, dtype=torch.float32, 
                                                                 non_blocking=True).div_(255.0)
                            else:
                                processed_obs[key] = value.to(device, non_blocking=True)
                        else:
                            processed_obs[key] = value
            else:
                processed_obs = obs.to(device, non_blocking=True) if hasattr(obs, 'to') else obs
            
            action = samples['action'].to(device, non_blocking=True) if hasattr(samples['action'], 'to') else samples['action']
            
            # 性能监控结束
            end_time = time.time()
            self._perf_stats['total_postprocess_time'] += (end_time - start_time)
            
            return {
                'obs': processed_obs,
                'action': action
            }
        else:
            # 原始格式，GPU端批量预处理优化
            with torch.cuda.stream(self._current_stream):
                # 异步数据传输，先传输再处理
                agent_pos = samples['state'].to(device, dtype=torch.float32, non_blocking=True)
                action = samples['action'].to(device, dtype=torch.float32, non_blocking=True)
                
                # 图像数据高效处理 - 安全的维度检查
                head_cam_raw = samples['head_camera']
                if head_cam_raw.dtype == torch.uint8:
                    # GPU端处理：传输+归一化，仅对4D张量使用channels_last
                    if head_cam_raw.ndim == 4:
                        head_cam = head_cam_raw.to(device, dtype=torch.float32, 
                                                  memory_format=torch.channels_last, 
                                                  non_blocking=True).div_(255.0)
                    else:
                        head_cam = head_cam_raw.to(device, dtype=torch.float32, 
                                                  non_blocking=True).div_(255.0)
                else:
                    if head_cam_raw.ndim == 4:
                        head_cam = head_cam_raw.to(device, memory_format=torch.channels_last, 
                                                  non_blocking=True)
                    else:
                        head_cam = head_cam_raw.to(device, non_blocking=True)
                        
            # 性能监控结束
            end_time = time.time()
            self._perf_stats['total_postprocess_time'] += (end_time - start_time)
                    
            return {
                'obs': {
                    'head_cam': head_cam,
                    'agent_pos': agent_pos,
                },
                'action': action
            }
    
    def get_performance_stats(self):
        """获取数据预处理性能统计"""
        if self._perf_stats['batch_count'] > 0:
            avg_postprocess = self._perf_stats['total_postprocess_time'] / self._perf_stats['batch_count']
            return {
                'avg_postprocess_time_ms': avg_postprocess * 1000,
                'total_batches': self._perf_stats['batch_count'],
                'total_postprocess_time': self._perf_stats['total_postprocess_time']
            }
        return {}

def _batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray, indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]
_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)

def batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray, indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    # 降低并行化阈值，提高并行处理效率
    if batch_size >= 8 and data.nbytes // batch_size >= 2 ** 15:  # 降低阈值
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
