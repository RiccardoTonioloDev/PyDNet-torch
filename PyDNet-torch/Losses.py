import torch
from torch import nn


def L1_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return torch.abs(img1 - img2)


def SSIM_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    C1 = 0.01**2
    C2 = 0.03**2

    mu_img1 = nn.functional.avg_pool2d(img1, 3, 1, 0)
    mu_img2 = nn.functional.avg_pool2d(img2, 3, 1, 0)

    sigma_img1 = nn.functional.avg_pool2d(img1**2, 3, 1, 0) - mu_img1**2
    sigma_img2 = nn.functional.avg_pool2d(img2**2, 3, 1, 0) - mu_img2**2
    sigma_img1img2 = nn.functional.avg_pool2d(img1 * img2, 3, 1, 0) - mu_img1 * mu_img2

    SSIM_n = (2 * mu_img1 * mu_img2 + C1) * (2 * sigma_img1img2 + C2)
    SSIM_d = (mu_img1**2 + mu_img2**2 + C1) * (sigma_img1 + sigma_img2 + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def L_ap(  # TODO:non penso vada bene
    img1: torch.Tensor, img2: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    Reconstruction error loss
    """
    SSIM = SSIM_loss(img1, img2).mean() * alpha
    L1 = L1_loss(img1, img2).mean() * (1 - alpha)

    return SSIM + L1


def gradient_x(img: torch.Tensor):
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx


def gradient_y(img: torch.Tensor):
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy


def disparity_smoothness(disparity: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    disparity_grad_x = gradient_x(disparity)
    disparity_grad_y = gradient_y(disparity)

    image_grad_x = gradient_x(image)
    image_grad_y = gradient_y(image)

    weights_x = torch.exp(-torch.mean(torch.abs(image_grad_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_grad_y), 1, keepdim=True))

    smoothness_x = disparity_grad_x * weights_x
    smoothness_y = disparity_grad_y * weights_y

    return smoothness_x + smoothness_y


def L_df(  # TODO:non penso vada bene
    disparity_left: torch.Tensor,
    image_left: torch.Tensor,
    disparity_right: torch.Tensor,
    image_right: torch.Tensor,
) -> torch.Tensor:
    """
    Disparity gradient loss
    """
    return disparity_smoothness()
