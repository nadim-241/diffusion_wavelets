#!/bin/python3
from inspect import isfunction
import torch
from torch import nn
import math


def exists(x):
    return x is not None


class UpsampleBlock(nn.Module):
    def __init__(self) -> None:
        super.__init__(self)


class DownsampleBlock(nn.Module):
    def __init__(self) -> None:
        super.__init__(self)


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
