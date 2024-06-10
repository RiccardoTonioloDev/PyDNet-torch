from typing import List, Tuple
import torch
from torch import nn


def L1_loss(est_batch: torch.Tensor, img_batch: torch.Tensor) -> torch.Tensor:
    """
    # L1 diffence
    `est_batch`: Tensor[B, C, H, W]
        The warped image batch
    `img_batch`: Tensor[B, C, H, W]
        The original image batch
    """
    return torch.abs(est_batch - img_batch)  # [B, C, H, W]


def SSIM_loss(est_batch: torch.Tensor, img_batch: torch.Tensor) -> torch.Tensor:
    """
    # Structural Similarity Index Measure
    `est_batch`: Tensor[B, C, H, W]
       The warped image batch
    `img_batch`: Tensor[B, C, H, W]
        The original image batch
    """
    C1 = 0.01**2
    C2 = 0.03**2

    mus_est = nn.functional.avg_pool2d(est_batch, 3, 1, 0)  # [B, C, H - 2, W - 2]
    mu_img = nn.functional.avg_pool2d(img_batch, 3, 1, 0)  # [B, C, H - 2, W - 2]

    sigma_est = nn.functional.avg_pool2d(est_batch**2, 3, 1, 0) - mus_est**2
    # [B, C, H - 2, W - 2]
    sigma_img = nn.functional.avg_pool2d(img_batch**2, 3, 1, 0) - mu_img**2
    # [B, C, H - 2, W - 2]
    sigma_est_img = (
        nn.functional.avg_pool2d(est_batch * img_batch, 3, 1, 0) - mus_est * mu_img
    )  # [B, C, H - 2, W - 2]

    SSIM_n = (2 * mus_est * mu_img + C1) * (2 * sigma_est_img + C2)
    # [B, C, H - 2, W - 2]
    SSIM_d = (mus_est**2 + mu_img**2 + C1) * (sigma_est + sigma_img + C2)
    # [B, C, H - 2, W - 2]

    SSIM = SSIM_n / SSIM_d  # [B, C, H - 2, W - 2]

    return torch.clamp((1 - SSIM) / 2, 0, 1)  # [B, C, H - 2, W - 2]


def L_ap(
    est_batch_pyramid: List[torch.Tensor],
    img_batch_pyramid: List[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    """
    # Reconstruction error loss (otherwise called image loss)
        `est_batch_pyramid`: List[Tensor[B, C, H, W]]
            Warped image batch at different scales
        `img_batch_pyramid`: List[Tensor[B, C, H, W]]
            Corresponding image batch at different scales
    """
    SSIM_pyramid = [
        SSIM_loss(est, img).mean() * alpha
        for est, img in zip(est_batch_pyramid, img_batch_pyramid)
    ]  #  List[Tensor[B]]
    #  Doing the SSIM on each image of the batch at the same time, for each resolution
    L1_pyramid = [
        L1_loss(est, img).mean() * (1 - alpha)
        for est, img in zip(est_batch_pyramid, img_batch_pyramid)
    ]  # List[Tensor[B]]
    #  Doing the L1 on each image of the batch at the same time, for each resolution
    weighted_sum = [ssim + l1 for ssim, l1 in zip(SSIM_pyramid, L1_pyramid)]
    #  Adding each SSIM value with the corresponding L1 value (values are already weighted)

    return sum(weighted_sum)  #  Tensor[Scalar]


def gradient_x(tensor: torch.Tensor):
    gx = tensor[:, :, :, :-1] - tensor[:, :, :, 1:]
    return gx


def gradient_y(tensor: torch.Tensor):
    gy = tensor[:, :, :-1, :] - tensor[:, :, 1:, :]
    return gy


def disparity_smoothness(
    disparity_batch: torch.Tensor, image_batch: torch.Tensor
) -> torch.Tensor:
    """
    Disparity smoothness
        `disparity_batch`: [B, 1, H, W] (the channels dimension is externally artificially added).
            Normally it would be [B, H, W] because disparities are single numbers in a specific
            pixel location.
            Calculated disparities for a specific side
        `image_batch`: [B, C, H, W]
            Images batch of the corresponding side
    """
    disparity_grad_x = torch.abs(gradient_x(disparity_batch))  # [B, 1, H, W - 1]
    disparity_grad_y = torch.abs(gradient_y(disparity_batch))  # [B, 1, H - 1, W]

    image_grad_x = gradient_x(image_batch)  # [B, C, H, W - 1]
    image_grad_y = gradient_y(image_batch)  # [B, C, H - 1, W]

    weights_x = torch.exp(
        -torch.mean(torch.abs(image_grad_x), 1, keepdim=True)
    )  # [B, 1, H, W - 1]
    # weights_y = torch.exp( # IN TEORIA NON VENGONO UTILIZZATI I PESI IN ALTEZZA NEL CALCOLO
    #    -torch.mean(torch.abs(image_grad_y), 1, keepdim=True)
    # )  # [B, 1, H - 1, W]

    smoothness_x = disparity_grad_x * weights_x  # [B, 1, H, W - 1]
    # smoothness_y = disparity_grad_y * weights_y  # [B, 1, H - 1, W]

    # smoothness_sum = (
    #    smoothness_x[:, :, :-1, :] + smoothness_y[:, :, :, :-1]
    # )  # [B, 1, H - 1, W - 1]

    # return torch.mean(torch.mean(smoothness_sum, dim=(2, 3)))  # Tensor[Scalar]
    return torch.mean(torch.abs(smoothness_x))  # Tensor[Scalar]


def L_df(
    disparity_batch_pyramid: List[torch.Tensor],
    image_batch_pyramid: List[torch.Tensor],
) -> torch.Tensor:
    """
    # Disparity gradient loss
        `disparity_batch_pyramid`: List[Tensor[B, 1, H, W]]
            Calculated disparities for a specific side (at various resolutions)
        `image_batch_pyramid`: List[[B, C, H, W]]
            Images batch of the corresponding side (at various resolutions)
    """
    return sum(
        [
            disparity_smoothness(disp, img) / 2**i
            for i, (disp, img) in enumerate(
                zip(disparity_batch_pyramid, image_batch_pyramid)
            )
        ]
    )


def bilinear_sampler_1d_h(
    img_batch: torch.Tensor, disp_batch: torch.Tensor
) -> torch.Tensor:
    """
    A bilinear sampler in 1D horizontally.
        `img_batch`: [B, C, H, W]
            Original image to warp by sampling.
        `disp_batch`: [B, 1, H, W]
            Disparities for sampling the original image to generate the warped one.
    """
    B, _, H, W = img_batch.shape

    # Create a mesh grid
    x_base = (
        torch.linspace(0, W - 1, W).view(1, 1, W).expand(B, H, W).to(disp_batch.device)
    )  # [B, H, W]
    y_base = (
        torch.linspace(0, H - 1, H).view(1, H, 1).expand(B, H, W).to(disp_batch.device)
    )  # [B, H, W]

    # Add disparity to x-coordinates
    x_shifts = x_base + disp_batch.squeeze(1)

    # Normalize grid coordinates to [-1, 1]
    x_norm = 2 * (x_shifts / (W - 1)) - 1
    y_norm = 2 * (y_base / (H - 1)) - 1

    # Stack and permute to create grid for grid_sample
    grid = torch.stack((x_norm, y_norm), dim=3)  # [B, H, W, 2]

    # Sample the image with bilinear interpolation
    output = nn.functional.grid_sample(img_batch, grid, align_corners=True)

    return output  # [B, C, H, W]


def generate_image_left(
    right_img_batch: torch.Tensor, right_disp_batch: torch.Tensor
) -> torch.Tensor:
    """
    It generates a left image, starting from the right one, using disparities to sample.
        `right_img_batch`: [B, C, H, W]
            Right images to warp into left ones using the corresponding disparities.
        `right_disp_batch`: [B, 1, H, W]
            The corresponding disparities to use for right images.
    """
    right_img_batch = right_img_batch.to(torch.float32)
    right_disp_batch = -right_disp_batch.to(torch.float32)
    return bilinear_sampler_1d_h(right_img_batch, right_disp_batch)


def generate_image_right(
    left_img_batch: torch.Tensor, left_disp_batch: torch.Tensor
) -> torch.Tensor:
    """
    It generates a right image, starting from the left one, using disparities to sample.
        `left_img_batch`: [B, C, H, W]
            Left images to warp into right ones using the corresponding disparities.
        `left_disp_batch`: [B, 1, H, W]
            The corresponding disparities to use for left images.
    """
    left_img_batch = left_img_batch.to(torch.float32)
    left_disp_batch = left_disp_batch.to(torch.float32)
    return bilinear_sampler_1d_h(left_img_batch, left_disp_batch)


def L_lr(
    disp_l_batch_pyramid: List[torch.Tensor],
    disp_r_batch_pyramid: List[torch.Tensor],
) -> torch.Tensor:
    """
    # Left-righ consistency loss
        `disp_l_batch_pyramid`: List[Tensor[B, 1, H, W]]
            Disparities for left images (at different scales).
        `disp_r_batch_pyramid`: List[Tensor[B, 1, H, W]]
            Disparities for right images (at different scales).
    """
    right_to_left_disp = [
        generate_image_left(disp_r, disp_l)  # [B, 1, H, W]
        for disp_r, disp_l in zip(disp_r_batch_pyramid, disp_l_batch_pyramid)
    ]
    left_to_right_disp = [
        generate_image_right(disp_l, disp_r)  # [B, 1, H, W]
        for disp_l, disp_r in zip(disp_l_batch_pyramid, disp_r_batch_pyramid)
    ]
    lr_left_loss = [
        torch.mean(torch.abs(rtl_disp - disp_l))
        for rtl_disp, disp_l in zip(right_to_left_disp, disp_l_batch_pyramid)
    ]
    lr_right_loss = [
        torch.mean(torch.abs(ltr_disp - disp_r))
        for ltr_disp, disp_r in zip(left_to_right_disp, disp_r_batch_pyramid)
    ]
    return sum(lr_left_loss + lr_right_loss)


def L_total(
    est_batch_pyramid_l: torch.Tensor,
    est_batch_pyramid_r: torch.Tensor,
    img_batch_pyramid_l: torch.Tensor,
    img_batch_pyramid_r: torch.Tensor,
    disp_batch_pyramid_l: torch.Tensor,
    disp_batch_pyramid_r: torch.Tensor,
    weight_SSIM=1,
    weight_ap=1,
    weight_lr=1,
    weight_df=0.1,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:  # TODO: controllare dimensioni dei tensori nel training
    """
    # Total loss
    It combines the various losses to obtain the total loss
        Images constructed by using disparities (at various resolutions):
            `est_batch_pyramid_l`: List[Tensor],
                Reconstructed (warped) left images
            `est_batch_pyramid_r`: List[Tensor],
                Reconstructed (warped) right images
        Images (at various resolutions):
            `img_batch_pyramid_l`: List[Tensor],
                Images in left batch.
            `img_batch_pyramid_r`: List[Tensor],
                Images in right batch.
        Disparities (at various resolutions)
            `disp_batch_pyramid_l`: List[Tensor],
                Disparities calculated on left batch.
            `disp_batch_pyramid_r`: List[Tensor],
                Disparities calculated on right batch.
    """
    L_ap_tot = L_ap(est_batch_pyramid_l, img_batch_pyramid_l, weight_SSIM) + L_ap(
        est_batch_pyramid_r, img_batch_pyramid_r, weight_SSIM
    )
    L_df_tot = L_df(
        disp_batch_pyramid_l,
        img_batch_pyramid_l,
    ) + L_df(
        disp_batch_pyramid_r,
        img_batch_pyramid_r,
    )
    L_lr_tot = L_lr(disp_batch_pyramid_l, disp_batch_pyramid_r)
    L_total = L_ap_tot * weight_ap + L_df_tot * weight_df + L_lr_tot * weight_lr
    return L_total, L_ap_tot, L_df_tot, L_lr_tot
