#!/bin/python3
import torch
from torch import nn
import math
import torch.nn.functional as F


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, affine=False):
        super.__init__()
        self.affine = affine
        self.step = nn.Linear(in_channels, (1 + affine) * out_channels)

    def forward(self, x, noise_embedding):
        batch = x.shape[0]
        if self.affine:
            gamma, beta = (
                self.step(noise_embedding).view(batch, -1, 1, 1).chunk(2, dim=1)
            )
            x = (1 + gamma) * x + beta
        else:
            x = x + self.step(noise_embedding).view(batch, -1, 1, 1)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, dim) -> None:
        super.__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.steps = nn.Sequential(self.upsample, self.conv)

    def forward(self, x):
        return self.steps(x)


class DownsampleBlock(nn.Module):
    def __init__(self, dim) -> None:
        super.__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = (
            torch.arange(count, dtype=noise_level.dtype, device=noise_level.device)
            / count
        )
        encoding = noise_level.unsqueeze(1) * torch.exp(
            -math.log(1e4) * step.unsqueeze(0)
        )
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class Swish(nn.Module):
    def __init__(self):
        super.__init__()

    def forward(self, x):
        return x + torch.sigmoid(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super.__init__()
        self.steps = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.steps(x)


class ResnetBlock(nn.Module):
    def __init__(
        self, dim, dim_out, num_groups=32, dropout=0, noise_emb_dim=None, affine=False
    ):
        super.__init__()
        self.block1 = Block(dim, dim_out, num_groups)
        self.block2 = Block(dim_out, dim_out, num_groups)

        self.noise_func = FeatureWiseAffine(noise_emb_dim, dim_out, affine)

        self.result_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity

    def forward(self, x, time_embed):
        y = self.block1(x)
        y = self.noise_func(y, time_embed)
        y = self.block2(y)
        return y + self.result_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, num_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(num_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlockWithAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        noise_embed_dim=None,
        num_groups=32,
        dropout=0,
        with_attn=False,
        affine=False,
    ):
        self.with_attn = with_attn
        self.resnet = ResnetBlock(
            dim, dim_out, num_groups, dropout, noise_embed_dim, affine
        )
        if with_attn:
            self.attn = SelfAttention(dim_out, num_groups=num_groups)

    def forward(self, x, time_embed):
        x = self.resnet(x, time_embed)
        if self.with_attn:
            x = self.attn(x)
        return x


class gNet(nn.Module):
    def __init__(
        self,
        in_channels=6,
        out_channels=3,
        init_channels=32,
        num_groups=32,
        layer_mults=[1, 2, 4, 8, 8],
        attention_res={8},
        blocks_per_layer=3,
        dropout=0,
        with_time_embed=True,
        image_size=128,
    ):
        super.__init__()
        self.initial_conv = nn.Conv2d(in_channels, init_channels, 3)
        if with_time_embed:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, init_channels * 4),
                Swish(),
                nn.Linear(init_channels * 4, init_channels),
            )
        else:
            self.time_mlp = None

        num_layers = len(layer_mults)
        pre_conv_channels = init_channels

        channels_list = [pre_conv_channels]
        current_size = image_size

        down_layers = [nn.Conv2d(in_channels, init_channels, kernel_size=3, padding=1)]

        for i in range(num_layers):
            is_last = i == num_layers - 1
            use_attention = current_size in attention_res
            channel_mult = layer_mults[i] * init_channels
            for _ in range(blocks_per_layer):
                down_layers.append(
                    ResnetBlockWithAttn(
                        pre_conv_channels,
                        channel_mult,
                        init_channels,
                        num_groups=num_groups,
                        dropout=dropout,
                        with_attn=use_attention,
                    )
                )
                channels_list.append(channel_mult)
                pre_conv_channels = channel_mult
            if not is_last:
                down_layers.append(DownsampleBlock(pre_conv_channels))
                channels_list.append(pre_conv_channels)
                current_size //= 2
        self.down_blocks = nn.ModuleList(down_layers)

        use_attention = current_size in attention_res
        self.middle_block = nn.ModuleList(
            ResnetBlockWithAttn(
                pre_conv_channels,
                pre_conv_channels,
                init_channels,
                num_groups,
                dropout,
                use_attention,
            ),
            ResnetBlockWithAttn(
                pre_conv_channels,
                pre_conv_channels,
                init_channels,
                num_groups,
                dropout,
                use_attention,
            ),
        )

        up_blocks = []
        for i in reversed(range(num_layers)):
            is_last = i == 0
            use_attention = current_size in attention_res
            channel_mult = init_channels * layer_mults[i]
            for _ in range(blocks_per_layer):
                up_blocks.append(
                    ResnetBlockWithAttn(
                        pre_conv_channels + channels_list.pop(),
                        channel_mult,
                        init_channels,
                        num_groups,
                        dropout,
                        use_attention,
                    )
                )
                pre_conv_channels = channel_mult
            if not is_last:
                up_blocks.append(UpsampleBlock(pre_conv_channels))

            self.up_blocks = nn.ModuleList(up_blocks)

            self.out_conv = Block(pre_conv_channels, out_channels, groups=num_groups)

        def forward(x, time):
            time_embed = self.time_mlp(time) if self.time_mlp is not None else None

            residuals = []
            for layer in self.down_blocks:
                if isinstance(layer, ResnetBlockWithAttn):
                    x = layer(x, time_embed)
                else:
                    x = layer(x)
                residuals.append(x)

            for layer in self.middle_block:
                if isinstance(layer, ResnetBlockWithAttn):
                    x = layer(x, time_embed)
                else:
                    x = layer(x)

            for layer in self.up_blocks:
                residual = residuals.pop()
                if isinstance(layer, ResnetBlockWithAttn):
                    x = F.interpolate(
                        x, size=(residual.size(2), residual.size(3)), mode="nearest"
                    )
                    x = torch.cat((x, residual), dim=1)
                    x = layer(x, time_embed)
                else:
                    x = layer(x)
            return self.out_conv(x)
