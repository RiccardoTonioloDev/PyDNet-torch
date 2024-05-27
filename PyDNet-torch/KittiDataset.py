from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import pandas as pd
from PIL import Image
import os
import random


class KittiDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        filenames_file_path: str,
        image_width: int,
        image_height: int,
        mode: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            data_path (str): Path to the dataset folder containing images.
            filenames_file_path (str): Path to the file containing image filenames.
            image_width (int): Width to resize the images.
            image_height (int): Height to resize the images.
            mode (str): can be "train" for training and "test" for testing
            transform (Optional[transforms.Compose]): an optional transformation to be applied to images
        """
        self.filenames_df = pd.read_csv(filenames_file_path, delim_whitespace=True)
        self.data_path = data_path
        self.image_tensorizer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (image_height, image_width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
            ]
        )
        self.mode = mode
        self.transform = transform

    def __len__(self) -> int:
        num_rows, _ = self.filenames_df.shape
        return num_rows

    def __getitem__(self, i: int) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        path_ith_row_left, path_ith_row_right = self.filenames_df.iloc[i]

        left_image_path = os.path.join(self.data_path, path_ith_row_left)
        right_image_path = os.path.join(self.data_path, path_ith_row_right)

        try:
            with Image.open(left_image_path) as left_image:
                left_image_tensor = self.image_tensorizer(left_image.convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Error loading left image: {left_image_path}. {e}")
        if self.transform:
            left_image_tensor = self.transform(left_image_tensor)

        # Checking for testing
        if self.mode != "train":
            return left_image_tensor

        try:
            with Image.open(right_image_path) as right_image:
                right_image_tensor = self.image_tensorizer(right_image.convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Error loading right image: {right_image_path}. {e}")
        if self.transform:
            right_image_tensor = self.transform(right_image_tensor)

        # Randomly flipping images
        if random.random() > 0.5:
            left_image_tensor = transforms.functional.hflip(left_image_tensor)
            right_image_tensor = transforms.functional.hflip(right_image_tensor)

        # Randomly augmenting images
        if random.random() > 0.5:
            left_image_tensor, right_image_tensor = self.augment_image_pair(
                left_image_tensor, right_image_tensor
            )

        return left_image_tensor, right_image_tensor

    def augment_image_pair(
        left_image: torch.Tensor, right_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shifting with random gamma
        gamma = random.uniform(0.8, 1.2)
        left_image = transforms.functional.adjust_gamma(left_image, gamma)
        right_image = transforms.functional.adjust_gamma(right_image, gamma)

        # Shifting with random brightness
        gamma = random.uniform(0.5, 2)
        left_image = transforms.functional.adjust_brightness(left_image, gamma)
        right_image = transforms.functional.adjust_brightness(right_image, gamma)

        # Shifting with random colors
        colors = [random.uniform(0.8, 1.2) for _ in range(3)]
        left_image = transforms.functional.adjust_saturation(left_image, colors[0])
        left_image = transforms.functional.adjust_contrast(left_image, colors[1])
        left_image = transforms.functional.adjust_hue(left_image, colors[2])
        right_image = transforms.functional.adjust_saturation(right_image, colors[0])
        right_image = transforms.functional.adjust_contrast(right_image, colors[1])
        right_image = transforms.functional.adjust_hue(right_image, colors[2])

        return left_image, right_image

    def make_dataloader(
        self,
        batch_size: int = 8,
        shuffle_batch: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        dataloader = DataLoader(
            self, batch_size, shuffle_batch, num_workers, pin_memory=pin_memory
        )
        return dataloader
