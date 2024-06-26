from typing import List
from Blocks.DownsizingBlock import DownsizingBlock
from Blocks.UpsizingBlock import UpsizingBlock
from Blocks.LevelConvolutionsBlock import LevelConvolutionsBlock
import torch.nn as nn
import torch.nn.functional as F
import torch
from Blocks.Xavier_initializer import xavier_init
from Config import Config


class Pydnet(nn.Module):
    def __init__(self, config: Config):
        super(Pydnet, self).__init__()

        # LEVEL 1
        conf = config.get_configuration()
        if conf.BlackAndWhite_processing:
            self.__downsizing_block_1 = DownsizingBlock(1, 16)
        else:
            self.__downsizing_block_1 = DownsizingBlock(3, 16)
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)

        # LEVEL 2
        self.__downsizing_block_2 = DownsizingBlock(16, 32)
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__downsizing_block_3 = DownsizingBlock(32, 64)
        self.__conv_block_3 = LevelConvolutionsBlock(64 + 8)

        # LEVEL 4
        self.__downsizing_block_4 = DownsizingBlock(64, 96)
        self.__conv_block_4 = LevelConvolutionsBlock(96 + 8)

        # LEVEL 5
        self.__downsizing_block_5 = DownsizingBlock(96, 128)
        self.__conv_block_5 = LevelConvolutionsBlock(128 + 8)

        # LEVEL 6
        self.__downsizing_block_6 = DownsizingBlock(128, 192)
        self.__conv_block_6 = LevelConvolutionsBlock(192)

        self.__upsizing_6 = UpsizingBlock(8, 8)
        self.__upsizing_5 = UpsizingBlock(8, 8)
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
        conv1 = self.__downsizing_block_1(x)  # [8, 16, 128, 256]
        conv2 = self.__downsizing_block_2(conv1)  # [8, 32, 64, 128]
        conv3 = self.__downsizing_block_3(conv2)  # [8, 64, 32, 64]
        conv4 = self.__downsizing_block_4(conv3)  # [8, 96, 16, 32]
        conv5 = self.__downsizing_block_5(conv4)  # [8, 128, 8, 16]
        conv6 = self.__downsizing_block_6(conv5)  # [8, 192, 4, 8]

        # LEVEL 6
        conv6b = self.__conv_block_6(conv6)
        disp6 = self.__level_activations(conv6b)

        conv6b = self.__upsizing_6(conv6b)

        # LEVEL 5
        concat5 = torch.cat((conv5, conv6b), 1)
        conv5b = self.__conv_block_5(concat5)
        disp5 = self.__level_activations(conv5b)

        conv5b = self.__upsizing_5(conv5b)

        # LEVEL 4
        concat4 = torch.cat((conv4, conv5b), 1)
        conv4b = self.__conv_block_4(concat4)
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
            disp5,
            disp6,
        ]

    @staticmethod
    def scale_pyramid(img_batch: torch.Tensor, num_scales: int = 6) -> torch.Tensor:
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


class Pydnet2(nn.Module):
    def __init__(self):
        super(Pydnet2, self).__init__()

        # LEVEL 1
        self.__downsizing_block_1 = DownsizingBlock(3, 16)
        self.__conv_block_1 = LevelConvolutionsBlock(16 + 8)

        # LEVEL 2
        self.__downsizing_block_2 = DownsizingBlock(16, 32)
        self.__conv_block_2 = LevelConvolutionsBlock(32 + 8)

        # LEVEL 3
        self.__downsizing_block_3 = DownsizingBlock(32, 64)
        self.__conv_block_3 = LevelConvolutionsBlock(64 + 8)

        # LEVEL 4
        self.__downsizing_block_4 = DownsizingBlock(64, 96)
        self.__conv_block_4 = LevelConvolutionsBlock(96)

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
        conv1 = self.__downsizing_block_1(x)  # [8, 16, 128, 256]
        conv2 = self.__downsizing_block_2(conv1)  # [8, 32, 64, 128]
        conv3 = self.__downsizing_block_3(conv2)  # [8, 64, 32, 64]
        conv4 = self.__downsizing_block_4(conv3)  # [8, 96, 16, 32]

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
