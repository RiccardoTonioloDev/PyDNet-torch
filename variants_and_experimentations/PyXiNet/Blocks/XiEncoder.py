from typing import List
from torch import nn
from .xinet import XiNet
import torch


class XiEncoder(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        alpha: float,
        gamma: float,
        num_layers: int,
        mid_channels: int,
        out_channels: int,
    ):
        super(XiEncoder, self).__init__()

        memo_output_channels = []
        self.xn = XiNet(
            input_shape=input_shape,
            memo_output_channels=memo_output_channels,
            alpha=alpha,
            gamma=gamma,
            num_layers=num_layers,
            base_filters=mid_channels,
        )
        self.deconv = nn.ConvTranspose2d(
            in_channels=memo_output_channels[0],
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("----------------------ENTERING-------------------------")
        print(x.size())
        x = self.xn(x)
        print(x.size())
        x = self.deconv(x)
        print(x.size())
        return self.activation(x)
