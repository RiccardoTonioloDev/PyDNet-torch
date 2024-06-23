import torch.nn as nn


class XiNetConvBlock(nn.Module):
    # TODO: Not finished
    def __init__(self, in_channels, out_channels, compression=4, attention=False):
        super(XiNetConvBlock, self).__init__()
        # Pointwise convolution to reduce dimensionality
        self.pointwise_conv1 = nn.Conv2d(
            in_channels, out_channels // compression, kernel_size=1, stride=1, padding=0
        )
        self.conv = nn.Conv2d(
            out_channels // compression,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pointwise_conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.relu6 = nn.ReLU6(inplace=True)
        self.attention = attention
        if self.attention:
            self.attention_layer = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, stride=1, padding=0
                ),  # Pointwise convolution
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),  # Standard 3x3 convolution
                nn.Sigmoid(),
            )

    def forward(self, x):
        out = self.pointwise_conv1(x)
        out = self.conv(out)
        out = self.relu6(out)
        if self.attention:
            att = self.attention_layer(out)
            out = out * att
        out = self.pointwise_conv2(out)
        return out
