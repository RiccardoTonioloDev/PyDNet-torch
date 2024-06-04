class Config:
    model_name = "PyDNet-V1-torch"
    data_path = "/media/Volume/KITTI/"
    filenames_file_training = "./filenames/"
    input_height = 256
    input_width = 512
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    image_width = 512
    image_height = 256
    shuffle_batch = True

    lr_loss_weight = 1
    # Left-right consistency weight in the total loss calculation

    alpha_image_loss = 0.85
    # Weight between SSIM and L1 in the image loss

    disp_gradient_loss_weight = 0.1
    # Disparity gradient weight in the total loss calculation

    output_direcotry = "./outputfiles/outputs/"
    # Output directory to test disparities
    checkpoint_path = "./outputfiles/checkpoints/"

    retrain = True
    # If True it retrains the model without using checkpoints
