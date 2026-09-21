import torch
import torch.nn as nn


class CrossAttention(nn.Module):
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