import torch.nn as nn
import torch
import math


class SequentialWithBroadcast(nn.Sequential):
    def forward(self, x, broadcasted_tensor):
        for module in self:
            x = module(x, broadcasted_tensor)
        return x


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

        self.relu6 = nn.ReLU6()

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
        out = self.pointwise_conv(x)
        # compression pointwise conv

        if broadcasted_tensor is not None:
            resizing_factor_height = broadcasted_tensor.shape[-2] // x.shape[-2]
            resizing_factor_width = broadcasted_tensor.shape[-1] // x.shape[-1]
            if (
                resizing_factor_width != 1 or resizing_factor_height != 1
            ):  # resize only if at least one dimension has to be resized
                broadcasted_tensor = torch.nn.functional.avg_pool2d(
                    broadcasted_tensor,
                    (resizing_factor_height, resizing_factor_width),
                    stride=1,
                    padding=(
                        1 if resizing_factor_height > 1 else 0,
                        1 if resizing_factor_width > 1 else 0,
                    ),
                )
            # resizing in width and height the broadcasted tensor

            broadcasted_tensor = self.broadcast_pointwise_conv(broadcasted_tensor)
            # resizing in channels dimension the broadcasted tensor

            out = out + broadcasted_tensor
            # broadcasting the compressed and scaled input tensor
        # applying broadcasted input if exists

        out = self.conv(out)
        # applying main convolution

        out = self.relu6(out)
        # function activation after main convolution

        out = self.attention_module(out)
        # applying attention module

        return out


class XiNet(nn.Module):
    def __init__(
        self,
        in_channels_first_block: int,
        out_channels_first_block: int,
        N: int = 1,
        D_i: int = 0,
        alpha: float = 0.45,
        gamma: float = 4,
        beta: float = 1.8,
    ):
        super(XiNet, self).__init__()
        net_out_channels = [math.floor(out_channels_first_block * alpha)] + [
            4
            * math.ceil(
                alpha
                * (2 ** (D_i - 2))
                * (1 + (((beta - 1) * i) / N))
                * out_channels_first_block
            )
            for i in range(1, N)
        ]
        net_in_channels = [math.floor(in_channels_first_block)] + net_out_channels[:-1]
        self.__XiNet_blocks = SequentialWithBroadcast(
            *[
                XiConv2D(in_c, out_c, gamma=gamma, broadcast_channels=3)
                for in_c, out_c in zip(net_in_channels, net_out_channels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__XiNet_blocks(x, x)


xinet1 = XiNet(3, 16, 4, 0, 0.4, 4, 2)
xinet2 = XiNet(16, 32, 4, 1, 0.4, 4, 2)
xinet3 = XiNet(32, 64, 4, 2, 0.4, 4, 2)
xinet4 = XiNet(64, 96, 4, 3, 0.4, 4, 2)
xinet5 = XiNet(96, 128, 4, 4, 0.4, 4, 2)
xinet6 = XiNet(128, 192, 4, 5, 0.4, 4, 2)


def count_params(model: nn.Module) -> int:
    return (sum(p.numel() for p in model.parameters() if p.requires_grad),)


print(
    "Number of parameters: ",
    count_params(xinet1)
    + count_params(xinet2)
    + count_params(xinet3)
    + count_params(xinet4)
    + count_params(xinet5)
    + count_params(xinet6),
)
