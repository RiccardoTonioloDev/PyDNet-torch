class ConfigCluster:
    model_name = "PyDNet-V1-torch"
    data_path = "/home/rtoniolo/Datasets/kitti/"
    filenames_file_training = (
        "/home/rtoniolo/Depth/PyDNet-torch/filenames/eigen_train_files_png.txt"
    )
    filenames_file_testing = (
        "/home/rtoniolo/Depth/PyDNet-torch/filenames/eigen_test_files_png.txt"
    )
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

    output_directory = "/home/rtoniolo/Depth/PyDNet-torch/outputfiles/outputs/"
    # Output directory for the disparities file and for cluster
    # logs (if you use slurm files)
    checkpoint_path = "/home/rtoniolo/Depth/PyDNet-torch/outputfiles/checkpoints/"
    # Directory to be used to store checkpoint files.

    retrain = True
    # If True it retrains the model without using checkpoints

    debug = False
    # Not used anymore but useful to enable certain code sections only when this
    # parameter is set to True.

    checkpoint_to_use_path = ""
    # Path of the checkpoint file to be used inside the model.

    disparities_to_use = ""
    # Path of the disparities file to be used for evaluations.
