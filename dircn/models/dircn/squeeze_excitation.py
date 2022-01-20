"""
Squeeze and Excitation block
"""

from typing import Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from .complex_conv import ComplexConv2d

class SqueezeExcitation(nn.Module):
    """
    Squeeze and excitation block based on:
    https://arxiv.org/abs/1709.01507
    ratio set at 1./16 as recommended by the paper
    """

    def __init__(self,
                 channels: int,
                 ratio: float = 1./16,
                 ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        squeezed_channels = max(1, int(channels*ratio))
        self.layer_1 = nn.Conv2d(in_channels=channels,
                                 out_channels=squeezed_channels,
                                 kernel_size=1,
                                 bias=True,
                                 )
        self.layer_2 = nn.Conv2d(in_channels=squeezed_channels,
                                 out_channels=channels,
                                 kernel_size=1,
                                 bias=True)
        # self.swish = MemoryEfficientSwish()
        self.act = nn.ReLU(inplace=True)
        # Could do this using linear layer aswell, but than we need to .view in forward
        # self.linear_1 = nn.Linear(in_features=channels, out_features=squeezed_channels, bias=True)
        # self.linear_2 = nn.Linear(in_features=squeezed_channels, out_features=channels, bias=True)

    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        x = torch.sigmoid(x) * inputs
        return x

class ComplexSqueezeExcitation(nn.Module):
    """
    Complex version of the Squeeze and excitation block based on:
    https://arxiv.org/abs/1709.01507
    ratio set at 1./16 as recommended by the paper
    """

    def __init__(self,
                 channels: int,
                 ratio: float = 1./16,
                 ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        squeezed_channels = max(1, int(channels*ratio))
        self.layer_1 = ComplexConv2d(
            in_channels=channels,
            out_channels=squeezed_channels,
            kernel_size=1,
            bias=True,
            )
        self.layer_2 = ComplexConv2d(
            in_channels=squeezed_channels,
            out_channels=channels,
            kernel_size=1,
            bias=True,
            )
        # self.swish = MemoryEfficientSwish()
        self.act = nn.ReLU(inplace=True)
        # Could do this using linear layer aswell, but than we need to .view in forward
        # self.linear_1 = nn.Linear(in_features=channels, out_features=squeezed_channels, bias=True)
        # self.linear_2 = nn.Linear(in_features=squeezed_channels, out_features=channels, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        adp_real = self.avg_pool(inputs[..., 0])
        adp_imag = self.avg_pool(inputs[..., 1])
        x = torch.stack([adp_real, adp_imag], dim=-1)
        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        x = torch.sigmoid(x)
        return x * inputs
        # real = x[..., 0]*inputs[..., 0] - x[..., 1]*inputs[..., 1]
        # imag = x[..., 1]*inputs[..., 0] + x[..., 0]*inputs[..., 1]
        # return torch.stack([real, imag], dim=-1)
