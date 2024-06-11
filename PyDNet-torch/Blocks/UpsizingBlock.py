from torch import nn
from .Xavier_initializer import xavier_init


class UpsizingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        self.__deconv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2
        )
        xavier_init(self.__deconv)
        self.__block = nn.Sequential(
            self.__deconv,
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.__block(x)
