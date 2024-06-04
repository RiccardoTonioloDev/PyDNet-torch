import torch
import wandb
import torch.optim as optim
from KittiDataset import KittiDataset
from Pydnet import Pydnet
from Config import Config
from Losses import L_total, generate_image_left, generate_image_right


def train():
    # Configurations
    config = Config().get_homelab_configuration()

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
    train_dataloader = train_dataset.make_dataloader(
        config.batch_size, config.shuffle_batch
    )

    # Model
    pydnet = Pydnet()
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

    # Training
    for epoch in range(num_epochs):
        for i, (left_img_batch, right_img_batch) in enumerate(train_dataloader):
            # Infering disparities based on left and right image batches
            left_disp_pyramid = pydnet(left_img_batch)
            right_disp_pyramid = pydnet(right_img_batch)
            # Creating pyramid of various resolutions for left and right image batches
            left_img_batch_pyramid = pydnet.scale_pyramid(left_img_batch)
            right_img_batch_pyramid = pydnet.scale_pyramid(right_img_batch)
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
            total_loss, img_loss, disp_grad_loss, lr_loss = L_total(
                est_batch_pyramid_left,
                est_batch_pyramid_right,
                left_img_batch_pyramid,
                right_img_batch_pyramid,
                left_disp_pyramid,
                right_disp_pyramid,
                weigh_SSIM=config.weight_SSIM,
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
                }
            )

        # Logging stats
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}, Learning rate: {config.learning_rate * lr_lambda(epoch)}"
        )

        # Saving checkpoint
        torch.save(pydnet.state_dict(), config.checkpoint_path)
