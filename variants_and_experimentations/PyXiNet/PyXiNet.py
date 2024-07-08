from typing import List, Literal
from Blocks.DownsizingBlock import DownsizingBlock
from Blocks.UpsizingBlock import UpsizingBlock
from Blocks.LevelConvolutionsBlock import LevelConvolutionsBlock
import torch.nn as nn
import torch.nn.functional as F
import torch
from Blocks.Xavier_initializer import xavier_init
from Blocks.Xi import channel_calculator
import math


class PyXiNet(nn.Module):
    def __init__(self, efficiency: Literal["S", "M", "L"]):
        super(PyXiNet, self).__init__()
        alpha = None
        beta = None
        gamma = None
        if efficiency == "S":
            alpha = 0.4
            beta = 2
            gamma = 4
        elif efficiency == "M":
            alpha = 0.425
            beta = 1.9
            gamma = 4
        elif efficiency == "L":
            alpha = 0.45
            beta = 2
            gamma = 4

        # LEVEL 1
        first_block_xi_out_channels = math.ceil(alpha * 16)
        self.__downsizing_block_1 = DownsizingBlock(
            3, 16, first_block_xi_out_channels, gamma
        )
        self.__conv_block_1 = LevelConvolutionsBlock(first_block_xi_out_channels + 8)

        # LEVEL 2
        second_block_xi_out_channels = channel_calculator(
            first_block_xi_out_channels, alpha, beta, 2, 1, 4
        )
        self.__downsizing_block_2 = DownsizingBlock(
            first_block_xi_out_channels, 32, second_block_xi_out_channels, gamma
        )
        self.__conv_block_2 = LevelConvolutionsBlock(second_block_xi_out_channels + 8)

        # LEVEL 3
        third_block_xi_out_channels = channel_calculator(
            first_block_xi_out_channels, alpha, beta, 3, 2, 4
        )
        self.__downsizing_block_3 = DownsizingBlock(
            second_block_xi_out_channels, 64, third_block_xi_out_channels, gamma
        )
        self.__conv_block_3 = LevelConvolutionsBlock(third_block_xi_out_channels + 8)

        # LEVEL 4
        fourth_block_xi_out_channels = channel_calculator(
            first_block_xi_out_channels, alpha, beta, 4, 3, 4
        )
        self.__downsizing_block_4 = DownsizingBlock(
            third_block_xi_out_channels, 96, fourth_block_xi_out_channels, gamma
        )
        self.__conv_block_4 = LevelConvolutionsBlock(fourth_block_xi_out_channels)

        self.__upsizing_4 = UpsizingBlock(8, 8)
        self.__upsizing_3 = UpsizingBlock(8, 8)
        self.__upsizing_2 = UpsizingBlock(8, 8)

        self.apply(xavier_init)

    def __level_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gives back the disparities for left images on the 0th channel and for the right images on the 1th channel.
            - `x`: torch.Tensor[B,C,W,H]

        Returns a torch.Tensor[B,2,W,H]
        """
        return 0.3 * torch.sigmoid(x[:, :2, :, :])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Level's starting blocks
        conv1 = self.__downsizing_block_1(x)  # [8, 7, 128, 256]
        conv2 = self.__downsizing_block_2(conv1)  # [8, 16, 64, 128]
        conv3 = self.__downsizing_block_3(conv2)  # [8, 36, 32, 64]
        conv4 = self.__downsizing_block_4(conv3)  # [8, 80, 16, 32]

        # LEVEL 4
        conv4b = self.__conv_block_4(conv4)
        disp4 = self.__level_activations(conv4b)

        conv4b = self.__upsizing_4(conv4b)

        # LEVEL 3
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.__conv_block_3(concat3)
        disp3 = self.__level_activations(conv3b)

        conv3b = self.__upsizing_3(conv3b)

        # LEVEL 2
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.__conv_block_2(concat2)
        disp2 = self.__level_activations(conv2b)

        conv2b = self.__upsizing_2(conv2b)

        # LEVEL 1
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.__conv_block_1(concat1)
        disp1 = self.__level_activations(conv1b)

        return [
            disp1,
            disp2,
            disp3,
            disp4,
        ]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 4) -> torch.Tensor:
        """
        It scales the batch of images to `num_scales` scales, everytime dividing by 2 the height and the width.
            - `img_batch`: torch.Tensor[B,C,H,W]
            - `num_scales`: int

        Returns a List[torch.Tensor[B,C,H,W]] with a length of `num_scales`.
        """
        scaled_imgs = []
        _, _, h, w = img_batch.shape
        for i in range(num_scales):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_img = nn.functional.interpolate(
                img_batch, size=(nh, nw), mode="area"
            )
            scaled_imgs.append(scaled_img)
        return scaled_imgs

    @staticmethod
    def upscale_img(
        img_tensor: torch.Tensor, new_2d_size: tuple[int, int] = (256, 512)
    ) -> torch.Tensor:
        """
        It upscales the `img_tensor` using bilinear interpolation to the specified size `new_2d_size`.
            - `img_tensor`: torch.Tensor[B,C,H,W];
            - `new_2d_size`: tuple(width,height).

        Returns a torch.Tensor[B,C,width, height].
        """
        return F.interpolate(
            img_tensor, size=new_2d_size, mode="bilinear", align_corners=True
        )


def count_params(model: nn.Module) -> int:
    return (sum(p.numel() for p in model.parameters() if p.requires_grad),)
