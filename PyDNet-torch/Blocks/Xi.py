import torch.nn as nn
import torch


class XiConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float = 1.0,
        gamma: int = 4,
        broadcast_channels: int = 0,
    ):
        super(XiConv2D, self).__init__()

        # Applying scaling factor to input and output channels
        in_channels = round(alpha * in_channels)
        out_channels = round(alpha * out_channels)

        # Choosing a compression that won't result in a 0 channels output
        gamma = min(out_channels, gamma)

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__gamma = gamma

        # Pointwise convolution to reduce dimensionality
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels // gamma, kernel_size=1, stride=1, padding=0
        )

        # Main convolution
        self.conv = nn.Conv2d(
            out_channels // gamma,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Broadcasting channel alignment pointwise convolution (only if something has to be broadcasted)
        self.broadcast_channels = broadcast_channels
        if broadcast_channels != 0:
            self.broadcast_pointwise_conv = nn.Conv2d(
                broadcast_channels,
                out_channels // gamma,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        self.attention_module = nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, broadcasted_tensor: torch.Tensor = None
    ) -> torch.Tensor:
        out = self.pointwise_conv1(x)
        if broadcasted_tensor is not None:
            resizing_factor = broadcasted_tensor.shape[2] // x.shape[2]
            resized_broadcasted_tensor = torch.nn.functional.avg_pool2d(
                broadcasted_tensor, resizing_factor, stride=1, padding=1
            )
            rechannelized_broadcasted_tensor = self.broadcast_pointwise_conv(
                resized_broadcasted_tensor
            )
            out = out + rechannelized_broadcasted_tensor
        out = self.conv(out)
        out = self.attention_module(out)
        return out


class XiNet(nn.Module):
    pass
