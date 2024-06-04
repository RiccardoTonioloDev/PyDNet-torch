from Blocks.DownsizingBlock import DownsizingBlock
from Blocks.LevelConvolutionsBlock import LevelConvolutionsBlock
import torch.nn as nn
import torch


class Pydnet(nn.Module):
    def __init__(self):
        super(Pydnet, self).__init__()

        # LEVEL 1
        self.downsizing_block_1 = DownsizingBlock(3, 16)
        self.conv_block_1 = LevelConvolutionsBlock(16)

        # LEVEL 2
        self.downsizing_block_2 = DownsizingBlock(16, 32)
        self.conv_block_2 = LevelConvolutionsBlock(32)

        # LEVEL 3
        self.downsizing_block_3 = DownsizingBlock(32, 64)
        self.conv_block_3 = LevelConvolutionsBlock(64)

        # LEVEL 4
        self.downsizing_block_4 = DownsizingBlock(64, 96)
        self.conv_block_4 = LevelConvolutionsBlock(96)

        # LEVEL 5
        self.downsizing_block_5 = DownsizingBlock(96, 128)
        self.conv_block_5 = LevelConvolutionsBlock(128)

        # LEVEL 6
        self.downsizing_block_6 = DownsizingBlock(128, 192)
        self.conv_block_6 = LevelConvolutionsBlock(192)

        self.level_activations = nn.Sigmoid()

        self.upsizing = nn.ConvTranspose2d(
            in_channels=8, out_channels=8, kernel_size=2, stride=2
        )

    def forward(self, x):
        # Level's starting blocks
        conv1 = self.downsizing_block_1(x)
        conv2 = self.downsizing_block_2(conv1)
        conv3 = self.downsizing_block_3(conv2)
        conv4 = self.downsizing_block_4(conv3)
        conv5 = self.downsizing_block_5(conv4)
        conv6 = self.downsizing_block_6(conv5)

        # LEVEL 6
        conv6b = self.conv_block_6(conv6)
        disp6 = self.level_activations(conv6b)

        conv6b = self.upsizing(conv6b)

        # LEVEL 5
        concat5 = torch.cat((conv5, conv6b), 1)
        conv5b = self.conv_block_5(concat5)
        disp5 = self.level_activations(conv5b)

        conv5b = self.upsizing(conv5b)

        # LEVEL 4
        concat4 = torch.cat((conv4, conv5b), 1)
        conv4b = self.conv_block_4(concat4)
        disp4 = self.level_activations(conv4b)

        conv4b = self.upsizing(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.conv_block_3(concat3)
        disp3 = self.level_activations(conv3b)

        conv3b = self.upsizing(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.conv_block_2(concat2)
        disp2 = self.level_activations(conv2b)

        conv2b = self.upsizing(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.conv_block_1(concat1)
        disp1 = self.level_activations(conv1b)

        return [disp1, disp2, disp3, disp4, disp5, disp6]

    def scale_pyramid(self, img_batch, num_scales):
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2**i
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs
