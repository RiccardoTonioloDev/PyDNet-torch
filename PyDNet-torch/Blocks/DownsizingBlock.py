from torch import nn
from .Xavier_initializer import xavier_init


class DownsizingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownsizingBlock, self).__init__()
        self.__first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.__second_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.block = nn.Sequential(  # TODO: ragionare sulla questione dello ZeroPad2D
            self.__first_conv,
            nn.LeakyReLU(0.2),
            self.__second_conv,
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)
