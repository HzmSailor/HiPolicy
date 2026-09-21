from typing import Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from termcolor import cprint
from diffusion_policy_3d.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy_3d.model.diffusion.positional_embedding import SinusoidalPosEmb



logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    def __init__(self, atten_dim, num_heads=8):
        super().__init__()
        self.atten_dim = atten_dim
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, atten_dim))
        self.cross_attention = nn.MultiheadAttention(embed_dim=atten_dim, num_heads=num_heads)

    def forward(self, x):

        # print("进入 CrossAttention 的 forward 函数：x.shape: ", x.shape)

        B, C, T = x.shape
        assert C*T//3 == self.atten_dim
        actual_horizon = T // 3

        x_long = x[:, :, :actual_horizon].reshape(B, -1).unsqueeze(1)
        x_mid = x[:, :, actual_horizon:actual_horizon*2].reshape(B, -1).unsqueeze(1)
        x_short = x[:, :, actual_horizon*2:].reshape(B, -1).unsqueeze(1)

        cls_tokens = self.cls_tokens.expand(B, -1, -1)

        # print("cls_tokens.shape: ", cls_tokens.shape)
        # print("x_long.shape: ", x_long.shape)
        # print("x_mid.shape: ", x_mid.shape)
        # print("x_short.shape: ", x_short.shape)

        input_tokens = torch.cat((cls_tokens, x_long, x_mid, x_short), dim=1)

        assert input_tokens.shape == (B, 4, self.atten_dim)

        global_feature = self.cross_attention(input_tokens, input_tokens, input_tokens)[0][:, 0, :].reshape(B, C, actual_horizon)

        return global_feature


class CrossAttention(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim):
        super().__init__()
        self.query_proj = nn.Linear(in_dim, out_dim)
        self.key_proj = nn.Linear(cond_dim, out_dim)
        self.value_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, x, cond):
        # x: [batch_size, t_act, in_dim]
        # cond: [batch_size, t_obs, cond_dim]

        # Project x and cond to query, key, and value
        query = self.query_proj(x)  # [batch_size, horizon, out_dim]
        key = self.key_proj(cond)   # [batch_size, horizon, out_dim]
        value = self.value_proj(cond)  # [batch_size, horizon, out_dim]


        # Compute attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, horizon, horizon]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, value)  # [batch_size, horizon, out_dim]
        
        return attn_output
    

class ConditionalResidualBlock1D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 global_feature_extractor=None,
                 use_global_feature=False,
                 kernel_size=3,
                 n_groups=8,
                 condition_type='film'):
        super().__init__()

        if use_global_feature == True:
            self.blocks = nn.ModuleList([
                Conv1dBlock(in_channels*2, out_channels*2, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels*2, out_channels, kernel_size, n_groups=n_groups),
            ])
        else:
            self.blocks = nn.ModuleList([
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ])


        self.condition_type = condition_type
        self.global_feature_extractor = global_feature_extractor
        self.use_global_feature = use_global_feature
        cond_channels = out_channels
        if condition_type == 'film': # FiLM modulation https://arxiv.org/abs/1709.07871
            # predicts per-channel scale and bias
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'add':
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'cross_attention_add':
            self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels)
        elif condition_type == 'cross_attention_film':
            cond_channels = out_channels * 2
            self.cond_encoder = CrossAttention(in_channels, cond_dim, cond_channels)
        elif condition_type == 'mlp_film':
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_dim),
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        else:
            raise NotImplementedError(f"condition_type {condition_type} not implemented")
        
        self.out_channels = out_channels
        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond_long=None, cond_mid=None, cond_short=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        actual_horizon = x.shape[2]//3
        x_original = x

        x_long = x[:, :, :actual_horizon]
        x_mid = x[:, :, actual_horizon:actual_horizon*2]
        x_short = x[:, :, actual_horizon*2:]

        # print("self.use_global_feature: ", self.use_global_feature)
        if self.use_global_feature == True:
            global_feature = self.global_feature_extractor(x)
        else:
            global_feature = None
        
        if self.use_global_feature == True:
            x_long = torch.cat([x_long, global_feature], dim=1)
            x_mid = torch.cat([x_mid, global_feature], dim=1)
            x_short = torch.cat([x_short, global_feature], dim=1)

        x = torch.cat([x_long, x_mid, x_short], dim=2)

        out = self.blocks[0](x)
        x_len = out.shape[1]//2
        # print("out.shape: ", out.shape)
        
        if self.use_global_feature == True:
            global_feature_long = out[:, x_len:, :actual_horizon]
            global_feature_mid = out[:, x_len:, actual_horizon:actual_horizon*2]
            global_feature_short = out[:, x_len:, actual_horizon*2:]
            # print("global_feature_long.shape: ", global_feature_long.shape)
            # print("global_feature_mid.shape: ", global_feature_mid.shape)
            # print("global_feature_short.shape: ", global_feature_short.shape)
            out_long = out[:, :x_len, :actual_horizon]
            out_mid = out[:, :x_len, actual_horizon:actual_horizon*2]
            out_short = out[:, :x_len, actual_horizon*2:]
        else:
            global_feature_long = None
            global_feature_mid = None
            global_feature_short = None

            out_long = out[:, :, :actual_horizon]
            out_mid = out[:, :, actual_horizon:actual_horizon*2]
            out_short = out[:, :, actual_horizon*2:]

        if cond_long is not None and cond_mid is not None and cond_short is not None:      
            if self.condition_type == 'film':
                embed_long = self.cond_encoder(cond_long)
                embed_mid = self.cond_encoder(cond_mid)
                embed_short = self.cond_encoder(cond_short)

                embed_long = embed_long.reshape(
                    embed_long.shape[0], 2, self.out_channels, 1)
                scale_long = embed_long[:,0,...]
                bias_long = embed_long[:,1,...]
                out_long = scale_long * out_long + bias_long

                embed_mid = embed_mid.reshape(
                    embed_mid.shape[0], 2, self.out_channels, 1)
                scale_mid = embed_mid[:,0,...]
                bias_mid = embed_mid[:,1,...]
                out_mid = scale_mid * out_mid + bias_mid

                embed_short = embed_short.reshape(
                    embed_short.shape[0], 2, self.out_channels, 1)
                scale_short = embed_short[:,0,...]
                bias_short = embed_short[:,1,...]
                out_short = scale_short * out_short + bias_short

                if self.use_global_feature == True:
                    out_long = torch.cat([out_long, global_feature_long], dim=1)
                    out_mid = torch.cat([out_mid, global_feature_mid], dim=1)
                    out_short = torch.cat([out_short, global_feature_short], dim=1)
                
                out = torch.cat([out_long, out_mid, out_short], dim=2)
                # print("out.shape: ", out.shape)
            elif self.condition_type == 'add':
                embed = self.cond_encoder(cond)
                out = out + embed
            elif self.condition_type == 'cross_attention_add':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1) # [batch_size, out_channels, horizon]
                out = out + embed
            elif self.condition_type == 'cross_attention_film':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'mlp_film':
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            else:
                raise NotImplementedError(f"condition_type {self.condition_type} not implemented")
        out = self.blocks[1](out)
        out = out + self.residual_conv(x_original)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        condition_type='film',
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        ):
        super().__init__()
        self.condition_type = condition_type
        
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition
        self.actual_horizon = 8
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, global_feature_extractor=None,
                    use_global_feature=False,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, global_feature_extractor=None,
                    use_global_feature=False,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim, global_feature_extractor=None,
                use_global_feature=False,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim, global_feature_extractor=None,
                use_global_feature=False,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
        ])

        down_modules = nn.ModuleList([])
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            if ind == 0:
                use_global_feature = True
            else:
                use_global_feature = False

            global_feature_extractor_1 = MultiHeadAttention(atten_dim=dim_in*self.actual_horizon//(2**ind), num_heads=8) if use_global_feature == True else None
            global_feature_extractor_2 = MultiHeadAttention(atten_dim=dim_out*self.actual_horizon//(2**ind), num_heads=8) if use_global_feature == True else None


            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, global_feature_extractor=global_feature_extractor_1,
                    use_global_feature=use_global_feature,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, global_feature_extractor=global_feature_extractor_2,
                    use_global_feature=use_global_feature,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim, global_feature_extractor=None,
                    use_global_feature=False,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim, global_feature_extractor=None,
                    use_global_feature=False,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond_long=None, global_cond_mid=None, global_cond_short=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        timestep_embed = self.diffusion_step_encoder(timesteps)
        if global_cond_long is not None and global_cond_mid is not None and global_cond_short is not None:
            if self.condition_type == 'cross_attention':
                timestep_embed = timestep_embed.unsqueeze(1).expand(-1, global_cond_long.shape[1], -1)
            global_feature_long = torch.cat([timestep_embed, global_cond_long], axis=-1)
            global_feature_mid = torch.cat([timestep_embed, global_cond_mid], axis=-1)
            global_feature_short = torch.cat([timestep_embed, global_cond_short], axis=-1)


        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature_long, global_feature_mid, global_feature_short)
            h_local.append(x)
            x = resnet2(local_cond, global_feature_long, global_feature_mid, global_feature_short)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if self.use_down_condition:
                x = resnet(x, global_feature_long, global_feature_mid, global_feature_short)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature_long, global_feature_mid, global_feature_short)
            else:
                x = resnet(x)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x)
            h.append(x)
            x = downsample(x)


        for mid_module in self.mid_modules:
            if self.use_mid_condition:
                x = mid_module(x, global_feature_long, global_feature_mid, global_feature_short)
            else:
                x = mid_module(x)


        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            if self.use_up_condition:
                x = resnet(x, global_feature_long, global_feature_mid, global_feature_short)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature_long, global_feature_mid, global_feature_short)
            else:
                x = resnet(x)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x)
            x = upsample(x)


        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x

