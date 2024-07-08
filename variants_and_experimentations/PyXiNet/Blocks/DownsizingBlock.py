from torch import nn
from .Xi import XiConv2D, channel_calculator
import math
import torch


class DownsizingBlock(nn.Module):
    def __init__(
        self,
        conv_in_channels: int,
        conv_out_channels: int,
        xi_conv_out_channels: int,
        gamma: float,
    ):
        super(DownsizingBlock, self).__init__()

        self.__block = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_in_channels,
                out_channels=conv_out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            XiConv2D(conv_out_channels, xi_conv_out_channels, gamma),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__block(x)
