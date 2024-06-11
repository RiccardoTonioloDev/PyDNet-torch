from torch import nn
from .Xavier_initializer import xavier_init


class LevelConvolutionsBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(LevelConvolutionsBlock, self).__init__()

        self.__first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=96,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # TODO: occhio al same
        xavier_init(self.__first_conv)
        self.__second_conv = nn.Conv2d(
            in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        xavier_init(self.__second_conv)
        self.__third_conv = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        xavier_init(self.__third_conv)
        self.__fourth_conv = nn.Conv2d(
            in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        xavier_init(self.__fourth_conv)

        self.__block = nn.Sequential(
            self.__first_conv,
            nn.LeakyReLU(0.2),
            self.__second_conv,
            nn.LeakyReLU(0.2),
            self.__third_conv,
            nn.LeakyReLU(0.2),
            self.__fourth_conv,
        )

    def forward(self, x):
        return self.__block(x)
