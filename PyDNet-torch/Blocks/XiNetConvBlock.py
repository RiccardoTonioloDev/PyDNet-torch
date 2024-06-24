import torch.nn as nn
import torch


class OptimizedConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float = 1.0,
        gamma: int = 4,
    ):
        super(OptimizedConv2D, self).__init__()

        # Applying scaling factor to input and output channels
        in_channels = round(alpha * in_channels)
        out_channels = round(alpha * out_channels)

        # Choosing a compression that won't result in a 0 channels output
        gamma = min(out_channels, gamma)

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__gamma = gamma

        # Pointwise convolution to reduce dimensionality
        self.pointwise_conv1 = nn.Conv2d(
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

        # Broadcasting channel alignment pointwise convolution
        # self.pointwise_conv2 = nn. # TODO: to finish

    def forward(
        self, x: torch.Tensor, broadcasted_tensor: torch.Tensor = None
    ) -> torch.Tensor:
        out = self.pointwise_conv1(x)
        if broadcasted_tensor is not None:
            out = out + broadcasted_tensor
        out = self.conv(out)
        return out

    def broadcast_pointwise_conv(self, x: torch.Tensor) -> torch.Tensor:
        pass  # TODO: to finish


class XiNetConvBlock(nn.Module):
    # TODO: Not finished
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        compression: int = 4,
    ):
        super(XiNetConvBlock, self).__init__()

        # Optimized convolutional block #1
        self.optimized_conv1 = OptimizedConv2D(in_channels, out_channels, compression)

        # Activation function for optimized convolution
        self.relu6 = nn.ReLU6(inplace=True)

        self.attention_layer = nn.Sequential(
            OptimizedConv2D(
                out_channels, out_channels, compression
            ),  # Standard 3x3 convolution
            nn.Sigmoid(),
        )

        self.broadcast_pointwise_conv = self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels // compression, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor):
        out = self.optimized_conv1.forward(x)
        out = self.relu6(out)
        att = self.attention_layer(out)
        out = out * att
        return out
