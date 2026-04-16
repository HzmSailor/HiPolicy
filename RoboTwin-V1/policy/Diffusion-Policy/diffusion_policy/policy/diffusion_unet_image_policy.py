from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import math
import time

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            ensemble=False,  # 新增ensemble参数
            no_hierarchical_cond=False,
            short_interval=1,
            mid_interval=2,
            long_interval=6,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * 3

        actual_horizon = 8
        # print("actual_horizon: ", actual_horizon)

        model = ConditionalUnet1D(
            input_dim=input_dim,
            actual_horizon=actual_horizon,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.n_obs_steps_use = 3
        self.actual_horizon = 8
        self.ensemble = ensemble  # 新增ensemble属性初始化
        self.no_hierarchical_cond = no_hierarchical_cond
        self.short_interval = short_interval
        self.mid_interval = mid_interval
        self.long_interval = long_interval

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None, global_cond_long=None, global_cond_mid=None, global_cond_short=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory= torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)


        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond_long=global_cond_long, global_cond_mid=global_cond_mid, global_cond_short=global_cond_short)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        
        model_device = next(self.parameters()).device
        nobs = dict_apply(nobs, lambda x: x.to(model_device, non_blocking=True))
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.actual_horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        
        # build input
        device = model_device
        dtype = self.dtype


        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # print(nobs['head_cam'].shape)
            # print(nobs['head_cam'][0, :7, 0, :5,:5])

            B = next(iter(nobs.values())).shape[0]
            T = self.actual_horizon

            # 向上取整
            n_obs_steps_use = self.n_obs_steps_use
            # print("n_obs_steps_use: ", n_obs_steps_use)
            

            # 取 short/mid/long 的时间索引
            short_indices = list(range(self.n_obs_steps - n_obs_steps_use, self.n_obs_steps))                       # [4,5,6]
            mid_indices   = list(range(self.n_obs_steps - n_obs_steps_use*self.mid_interval + self.mid_interval -1, self.n_obs_steps, self.mid_interval))[:n_obs_steps_use]    # [2,4,6]
            long_indices  = list(range(self.n_obs_steps - n_obs_steps_use*self.long_interval + self.long_interval -1, self.n_obs_steps, self.long_interval))[:n_obs_steps_use]    # [0,4,6]

            assert len(short_indices) == n_obs_steps_use
            assert len(mid_indices)   == n_obs_steps_use
            assert len(long_indices)  == n_obs_steps_use

            # 分别处理图像观测和低维观测
            this_nobs_short = {}
            this_nobs_mid = {}
            this_nobs_long = {}
            
            # for key, x in nobs.items():
            #     if key in self.obs_encoder.low_dim_keys:
            #         # 低维观测只取最后一帧
            #         this_nobs_short[key] = x[:, short_indices[-1]]  # 只取最后一帧
            #         this_nobs_mid[key] = x[:, mid_indices[-1]]
            #         this_nobs_long[key] = x[:, long_indices[-1]]
            #     else:
            #         # 图像观测保持原有逻辑
            #         this_nobs_short[key] = x[:, short_indices, ...].reshape(-1, *x.shape[2:])
            #         this_nobs_mid[key] = x[:, mid_indices, ...].reshape(-1, *x.shape[2:])
            #         this_nobs_long[key] = x[:, long_indices, ...].reshape(-1, *x.shape[2:])

            this_nobs_short = dict_apply(nobs, lambda x: x[:, short_indices, ...].reshape(-1, *x.shape[2:]))
            this_nobs_mid   = dict_apply(nobs, lambda x: x[:, mid_indices,   ...].reshape(-1, *x.shape[2:]))
            this_nobs_long  = dict_apply(nobs, lambda x: x[:, long_indices,  ...].reshape(-1, *x.shape[2:]))

            # 去掉 mid long 的 qpos 观测
            # this_nobs_mid['agent_pos'][:, [6, 13]] = -1
            # this_nobs_long['agent_pos'][:, [6, 13]] = -1
           

            nobs_features_long = self.obs_encoder(this_nobs_long)
            nobs_features_mid = self.obs_encoder(this_nobs_mid)
            nobs_features_short = self.obs_encoder(this_nobs_short)

            # reshape back to B, Do
            global_cond_long = nobs_features_long.reshape(B, -1)
            global_cond_mid = nobs_features_mid.reshape(B, -1)
            global_cond_short = nobs_features_short.reshape(B, -1)

            # print("global_cond_long.shape: ", global_cond_long.shape)
            # print("global_cond_mid.shape: ", global_cond_mid.shape)
            # print("global_cond_short.shape: ", global_cond_short.shape)

            # empty data for action
            cond_data = torch.zeros(size=(B, T*3, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            # print("cond_data_long.shape: ", cond_data_long.shape)
            # print("cond_data_mid.shape: ", cond_data_mid.shape)
            # print("cond_data_short.shape: ", cond_data_short.shape)

            # print("cond_mask_long.shape: ", cond_mask_long.shape)  # (B, 8, 14)
            # print("cond_mask_mid.shape: ", cond_mask_mid.shape)
            # print("cond_mask_short.shape: ", cond_mask_short.shape)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        if self.no_hierarchical_cond:
            nsample = self.conditional_sample(
                cond_data,
                cond_mask,
                local_cond=local_cond,
                global_cond_long=global_cond_short,
                global_cond_mid=global_cond_short,
                global_cond_short=global_cond_short,
                **self.kwargs)
        else:
            nsample = self.conditional_sample(
                cond_data,
                cond_mask,
                local_cond=local_cond,
                global_cond_long=global_cond_long,
                global_cond_mid=global_cond_mid,
                global_cond_short=global_cond_short,
                **self.kwargs)
        
        # unnormalize prediction
        naction_pred_long = nsample[:, :self.actual_horizon, :Da]
        naction_pred_mid = nsample[:, self.actual_horizon:self.actual_horizon*2, :Da]
        naction_pred_short = nsample[:, self.actual_horizon*2:, :Da]
        action_pred_long = self.normalizer['action'].unnormalize(naction_pred_long)
        action_pred_mid = self.normalizer['action'].unnormalize(naction_pred_mid)
        action_pred_short = self.normalizer['action'].unnormalize(naction_pred_short)

        action_pred = action_pred_short
        # get action
        # print("self.ensemble: ", self.ensemble)
        action = action_pred_short
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        model_device = next(self.parameters()).device
        self.normalizer.to(model_device)

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
 
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # print("ctrl_pts_long:", batch['ctrl_pts_long'][0, ...])

        local_cond = None
        global_cond = None
        trajectory = nactions

        second_dim_size = trajectory.shape[1]  # 29
        
        indices_long = torch.arange(self.n_obs_steps, second_dim_size, self.long_interval)  
        # print("indices_long: ", indices_long)
        trajectory_long = trajectory[:, indices_long, :]  # 形状变为 (32, 8, 14)

        max_len = indices_long.shape[0]
        # print("max_len: ", max_len)
        indices_mid = torch.arange(self.n_obs_steps, second_dim_size, self.mid_interval)[:max_len] 
        # print("indices_mid: ", indices_mid)
        trajectory_mid = trajectory[:, indices_mid, :]  # 形状变为 (32, 8, 14)
        trajectory_short = trajectory[:, self.n_obs_steps:max_len+self.n_obs_steps, :]
        merge_trajectory = torch.cat([trajectory_long, trajectory_mid, trajectory_short], dim=1)

        cond_data = merge_trajectory
        
        if self.obs_as_global_cond:
            B, T = next(iter(nobs.values())).shape[:2]

            n_obs_steps_use = self.n_obs_steps_use
            # print("n_obs_steps_use: ", n_obs_steps_use)

            # 取 short/mid/long 的时间索引
            short_indices = list(range(self.n_obs_steps - n_obs_steps_use, self.n_obs_steps))                       # [4,5,6]
            mid_indices   = list(range(self.n_obs_steps - n_obs_steps_use*self.mid_interval + self.mid_interval -1, self.n_obs_steps, self.mid_interval))[:n_obs_steps_use]    # [2,4,6]
            long_indices  = list(range(self.n_obs_steps - n_obs_steps_use*self.long_interval + self.long_interval -1, self.n_obs_steps, self.long_interval))[:n_obs_steps_use]    # [0,4,6]

            # print("short_indices: ", short_indices)
            # print("mid_indices: ", mid_indices)
            # print("long_indices: ", long_indices)


            assert len(short_indices) == n_obs_steps_use
            assert len(mid_indices)   == n_obs_steps_use
            assert len(long_indices)  == n_obs_steps_use

            model_device = next(self.parameters()).device
            
            # 分别处理图像观测和低维观测
            this_nobs_short = {}
            this_nobs_mid = {}
            this_nobs_long = {}
            
            this_nobs_short = dict_apply(nobs, lambda x: x[:, short_indices, ...].reshape(-1, *x.shape[2:]))
            this_nobs_mid   = dict_apply(nobs, lambda x: x[:, mid_indices,   ...].reshape(-1, *x.shape[2:]))
            this_nobs_long  = dict_apply(nobs, lambda x: x[:, long_indices,  ...].reshape(-1, *x.shape[2:]))

            # for key, x in nobs.items():
            #     if key in self.obs_encoder.low_dim_keys:
            #         # 低维观测只取最后一帧
            #         this_nobs_short[key] = x[:, short_indices[-1]]  # 只取最后一帧
            #         this_nobs_mid[key] = x[:, mid_indices[-1]]
            #         this_nobs_long[key] = x[:, long_indices[-1]]
            #     else:
            #         # 图像观测保持原有逻辑
            #         this_nobs_short[key] = x[:, short_indices, ...].reshape(-1, *x.shape[2:])
            #         this_nobs_mid[key] = x[:, mid_indices, ...].reshape(-1, *x.shape[2:])
            #         this_nobs_long[key] = x[:, long_indices, ...].reshape(-1, *x.shape[2:])
            
            this_nobs_short = dict_apply(this_nobs_short, lambda x: x.to(model_device, non_blocking=True))
            this_nobs_mid   = dict_apply(this_nobs_mid, lambda x: x.to(model_device, non_blocking=True))
            this_nobs_long  = dict_apply(this_nobs_long, lambda x: x.to(model_device, non_blocking=True))
            
            # print("this_nobs.keys(): ", this_nobs_short.keys())
            # print("this_nobs_short.shape: ", this_nobs_short['agent_pos'].shape)
            # print("this_nobs_mid.shape: ", this_nobs_mid['agent_pos'].shape)
            # print("this_nobs_long.shape: ", this_nobs_long['agent_pos'].shape)
            
            # 去掉 mid long 的 qpos 观测
            # this_nobs_long['agent_pos'][:, [6, 13]] = -1
            # this_nobs_mid['agent_pos'][:, [6, 13]] = -1
            
            nobs_features_long = self.obs_encoder(this_nobs_long)
            nobs_features_mid = self.obs_encoder(this_nobs_mid)
            nobs_features_short = self.obs_encoder(this_nobs_short)
            
            
            # print("nobs_features_long.shape: ", nobs_features_long.shape)
            # print("nobs_features_mid.shape: ", nobs_features_mid.shape)
            # print("nobs_features_short.shape: ", nobs_features_short.shape)

            # reshape back to B, Do
            global_cond_long = nobs_features_long.reshape(batch_size, -1)
            global_cond_mid = nobs_features_mid.reshape(batch_size, -1)
            global_cond_short = nobs_features_short.reshape(batch_size, -1)
            

        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, self.horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory_long = cond_data.detach()

        condition_mask = self.mask_generator(merge_trajectory.shape)
        condition_mask = condition_mask.to(merge_trajectory.device, non_blocking=True)
        
        noise = torch.randn(merge_trajectory.shape, device=merge_trajectory.device)
        
        bsz = merge_trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=merge_trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            merge_trajectory, noise, timesteps)
        
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        device = merge_trajectory.device

        if self.no_hierarchical_cond:
            # print("进入无分层条件模式！")
            global_cond_short = global_cond_short.to(device, non_blocking=True) if global_cond_short is not None else None
            pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, 
                global_cond_long=global_cond_short, global_cond_mid=global_cond_short, 
                global_cond_short=global_cond_short)
        else:
            # print("进入有分层条件模式！")
            global_cond_long = global_cond_long.to(device, non_blocking=True) if global_cond_long is not None else None
            global_cond_mid = global_cond_mid.to(device, non_blocking=True) if global_cond_mid is not None else None
            global_cond_short = global_cond_short.to(device, non_blocking=True) if global_cond_short is not None else None
            pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, 
                global_cond_long=global_cond_long, global_cond_mid=global_cond_mid, 
                global_cond_short=global_cond_short)
        
        
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = merge_trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)

        # 把 mid long 的 gripper loss 全部替换为 short 的 gripper loss
        # loss[:, :8, [6, 13]] = loss[:, 16:24, [6, 13]]
        # loss[:, 8:16, [6, 13]] = loss[:, 16:24, [6, 13]]
    
        gripper_loss = loss[:, 16:24, [6, 13]]
        gripper_loss = gripper_loss.mean()
    
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss, gripper_loss
