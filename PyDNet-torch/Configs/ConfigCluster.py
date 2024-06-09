class ConfigHomeLab:
    model_name = "PyDNet-V1-torch"
    data_path = "/home/rtoniolo/Datasets/kitti/"
    filenames_file_training = "./filenames/eigen_train_files_png.txt"
    filenames_file_testing = "./filenames/eigen_test_files_png.txt"
    input_height = 256
    input_width = 512
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    image_width = 512
    image_height = 256
    shuffle_batch = True

    weight_lr = 1
    # Left-right consistency weight in the total loss calculation

    weight_ap = 1
    # Reconstruction error loss weight in the total loss calculation

    weight_df = 0.1
    # Disparity gradient weight in the total loss calculation

    weight_SSIM = 0.85
    # Weight between SSIM and L1 in the image loss

    output_direcotry = "./outputfiles/outputs/"
    # Output directory to test disparities
    checkpoint_path = "./outputfiles/checkpoints/"

    retrain = True
    # If True it retrains the model without using checkpoints
