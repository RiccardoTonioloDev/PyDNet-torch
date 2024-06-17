import os
import time
from typing import List, Literal
import torch
import torch.utils
import torch.utils.data
import wandb
import torch.optim as optim
from KittiDataset import KittiDataset
from Pydnet import Pydnet
from Config import Config
from Losses import L_total, generate_image_left, generate_image_right
from Configs.ConfigHomeLab import ConfigHomeLab
from Configs.ConfigCluster import ConfigCluster

import matplotlib.pyplot as plt
import numpy as np


def imshow(left_img, est_img, right_img, disparity):
    left_img = left_img / 2 + 0.5  # de-normalizza
    est_img = est_img / 2 + 0.5  # de-normalizza
    right_img = right_img / 2 + 0.5  # de-normalizza

    npimg1 = (
        left_img.detach().cpu().numpy()
    )  # Sposta l'immagine su CPU e convertila in numpy
    npimg2 = (
        est_img.detach().cpu().numpy()
    )  # Sposta l'immagine su CPU e convertila in numpy
    npimg3 = (
        right_img.detach().cpu().numpy()
    )  # Sposta l'immagine su CPU e convertila in numpy
    np_disp = (
        disparity.detach().cpu().numpy()
    )  # Sposta la disparità su CPU e convertila in numpy

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    axs[0].imshow(np.transpose(npimg1, (1, 2, 0)))
    axs[0].set_title("Left Image")
    axs[1].imshow(np.transpose(npimg2, (1, 2, 0)))
    axs[1].set_title("Estimated Image")
    axs[2].imshow(np.transpose(npimg3, (1, 2, 0)))
    axs[2].set_title("Right Image")
    axs[3].imshow(np_disp[0], cmap="viridis", vmin=0, vmax=1)
    axs[3].set_title("Disparity Heatmap")
    plt.show()


def train(env: Literal["HomeLab", "Cluster"]):
    # Configurations
    config = Config(env).get_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuring wandb
    wandb.init(
        project=config.model_name,
        config={
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
        },
    )

    # Data
    train_dataset = KittiDataset(
        config.data_path,
        config.filenames_file_training,
        config.image_width,
        config.image_height,
    )
    test_dataset = KittiDataset(
        config.data_path,
        config.filenames_file_testing,
        config.image_width,
        config.image_height,
    )
    train_dataloader = train_dataset.make_dataloader(
        config.batch_size, config.shuffle_batch
    )

    # Model
    pydnet = Pydnet().to(device)
    num_of_params = sum(p.numel() for p in pydnet.parameters())
    print("Total number of parameters: ", num_of_params)
    # Optimizer
    optimizer = optim.Adam(
        pydnet.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8
    )

    # Dynamic learning rate configuration
    num_epochs = config.num_epochs
    lr_lambda = lambda epoch: (
        0.5 ** (1 + int(epoch >= 0.8 * num_epochs)) if epoch >= 0.6 * num_epochs else 1
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * num_epochs

    start_time = time.time() / 3600  # in hours

    min_loss = torch.Tensor([0]).to(device)

    # Training
    for epoch in range(num_epochs):
        pydnet.train()
        for i, (left_img_batch, right_img_batch) in enumerate(train_dataloader):
            left_img_batch, right_img_batch = left_img_batch.to(
                device
            ), right_img_batch.to(device)

            # Model input
            # model_input = torch.cat([left_img_batch, right_img_batch], 1)
            # Infering disparities based on left and right image batches
            model_output: List[torch.Tensor] = pydnet(left_img_batch)
            left_disp_pyramid = [mo[:, 0, :, :].unsqueeze(1) for mo in model_output]
            right_disp_pyramid = [mo[:, 1, :, :].unsqueeze(1) for mo in model_output]

            # Infering disparities based on left and right image batches
            # left_disp_pyramid = pydnet(left_img_batch)
            # right_disp_pyramid = pydnet(right_img_batch)

            # Creating pyramid of various resolutions for left and right image batches
            left_img_batch_pyramid = pydnet.scale_pyramid(
                left_img_batch, 6
            )  # [B, C, H, W]
            right_img_batch_pyramid = pydnet.scale_pyramid(
                right_img_batch, 6
            )  # [B, C, H, W]

            # Using disparities to generate corresponding left and right warped image batches (at various resolutions)
            est_batch_pyramid_left = [
                generate_image_left(img, disp)
                for img, disp in zip(right_img_batch_pyramid, left_disp_pyramid)
            ]
            est_batch_pyramid_right = [
                generate_image_right(img, disp)
                for img, disp in zip(left_img_batch_pyramid, right_disp_pyramid)
            ]
            # Calculating the loss based on the total loss function
            total_loss, img_loss, disp_grad_loss, lr_loss = L_total(
                est_batch_pyramid_left,
                est_batch_pyramid_right,
                left_img_batch_pyramid,
                right_img_batch_pyramid,
                left_disp_pyramid,
                right_disp_pyramid,
                weight_SSIM=config.weight_SSIM,
                weight_ap=config.weight_ap,
                weight_lr=config.weight_lr,
                weight_df=config.weight_df,
            )
            # Doing the forward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "image_loss": img_loss.item(),
                    "disp_gradient_loss": disp_grad_loss.item(),
                    "lr_loss": lr_loss.item(),
                    "total_loss": total_loss.item(),
                    "avg_disparity_in_first_level": left_disp_pyramid[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .mean(),
                }
            )
            steps_done = i + epoch * steps_per_epoch

            if False and steps_done % 10 == 0 and i != 0:
                [
                    imshow(
                        left_img_batch_pyramid[i][0],
                        est_batch_pyramid_left[i][0],
                        right_img_batch_pyramid[i][0],
                        left_disp_pyramid[i][0],
                    )
                    for i in range(len(left_img_batch_pyramid))
                ]
            if steps_done % 100 == 0 and i != 0:
                elapsed_time = (time.time() / 3600) - start_time  # in hours
                steps_to_do = total_steps - steps_done
                time_remaining = (elapsed_time / steps_done) * steps_to_do
                # Logging stats
                print(
                    f"Epoch [{epoch+1}/{num_epochs}]| Steps: {steps_done}| Loss: {total_loss.item():.4f}| Learning rate: {config.learning_rate * lr_lambda(epoch)}| Elapsed time: {elapsed_time:.2f}h| Time to finish: ~{time_remaining}h|"
                )

        eval_loss = eval(test_dataset, config, pydnet, device)
        if min_loss > eval_loss:
            min_loss = eval_loss

            state = {
                "epoch": epoch,
                "model_state_dict": pydnet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss,
            }
            # Saving checkpoint
            torch.save(
                state,
                os.path.join(config.checkpoint_path, f"checkpoint_e{epoch}.pth.tar"),
            )
        print(f"Test loss: {eval_loss.item()}")


def eval(
    dataset: KittiDataset,
    config: ConfigHomeLab | ConfigCluster,
    pydnet: Pydnet,
    device: torch.device,
) -> torch.Tensor:
    pydnet.eval()
    test_dataloader = dataset.make_dataloader(config.batch_size, False)
    total_loss_tower = []
    with torch.no_grad():
        for i, (left_img_batch, right_img_batch) in enumerate(test_dataloader):
            left_img_batch, right_img_batch = left_img_batch.to(
                device
            ), right_img_batch.to(device)
            # Infering disparities based on left and right image batches
            left_disp_pyramid = pydnet(left_img_batch)  # [B, 1, H, W]
            right_disp_pyramid = pydnet(right_img_batch)  # [B, 1, H, W]

            # Creating pyramid of various resolutions for left and right image batches
            left_img_batch_pyramid = pydnet.scale_pyramid(
                left_img_batch, 6
            )  # [B, C, H, W]
            right_img_batch_pyramid = pydnet.scale_pyramid(
                right_img_batch, 6
            )  # [B, C, H, W]

            # Using disparities to generate corresponding left and right warped image batches (at various resolutions)
            est_batch_pyramid_left = [
                generate_image_left(img, disp)
                for img, disp in zip(right_img_batch_pyramid, right_disp_pyramid)
            ]
            est_batch_pyramid_right = [
                generate_image_right(img, disp)
                for img, disp in zip(left_img_batch_pyramid, left_disp_pyramid)
            ]
            # Calculating the loss based on the total loss function
            total_loss, _, _, _ = L_total(
                est_batch_pyramid_left,
                est_batch_pyramid_right,
                left_img_batch_pyramid,
                right_img_batch_pyramid,
                left_disp_pyramid,
                right_disp_pyramid,
                weight_SSIM=config.weight_SSIM,
                weight_ap=config.weight_ap,
                weight_lr=config.weight_lr,
                weight_df=config.weight_df,
            )
            total_loss_tower.append(total_loss)
        tower_len = len(total_loss_tower)
        avg_loss = sum(total_loss_tower) / tower_len
        return avg_loss
