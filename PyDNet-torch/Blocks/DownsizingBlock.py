from torch import nn


class DownsizingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownsizingBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)
