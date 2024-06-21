class ConfigHomeLab:
    model_name: str = "PyDNet-V1-torch"
    data_path: str = "/media/riccardo-toniolo/Volume/KITTI/"
    filenames_file_training: str = "./filenames/eigen_train_files_png.txt"
    filenames_file_testing: str = "./filenames/eigen_test_files_png.txt"
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    image_width: int = 512
    image_height: int = 256
    shuffle_batch: bool = True

    weight_lr: float = 1
    # Left-right consistency weight in the total loss calculation

    weight_ap: float = 1
    # Reconstruction error loss weight in the total loss calculation

    weight_df: float = 0.1
    # Disparity gradient weight in the total loss calculation

    weight_SSIM: float = 0.85
    # Weight between SSIM and L1 in the image loss

    output_directory: str = "./outputfiles/outputs/"
    # Output directory for the disparities file and for cluster
    # logs (if you use slurm files)
    checkpoint_path: str = "./outputfiles/checkpoints/"
    # Directory to be used to store checkpoint files.

    retrain: bool = True
    # If True it retrains the model without using checkpoints

    debug: bool = True
    # Not used anymore but useful to enable certain code sections only when this
    # parameter is set to True.

    checkpoint_to_use_path: str = "./outputfiles/checkpoints/PyDNet-torch_50.pth.tar"
    # Path of the checkpoint file to be used inside the model.

    disparities_to_use: str = "./outputfiles/outputs/disparities.npy"
    # Path of the disparities file to be used for evaluations.
