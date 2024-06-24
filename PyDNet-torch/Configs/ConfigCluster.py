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
    num_epochs = 200
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

    ########################## EXPERIMENTS PARAMETERS ##########################
    # This parameters are only used to test the behaviour of the model and the #
    # training using specific conditions. All of them are meant to be turned   #
    # off for PyDNet to be trained as the original paper meant.                #
    ############################################################################

    HSV_processing: bool = False
    # It means that images will be processed in HSV format instead of RGB.

    BlackAndWhite_processing: bool = False
    # It means that images will be processed only in the gray scale (single channel).

    VerticalFlipAugmentation: bool = False
    # It means that images will have a 50% chance of being flipped upside-down.

    KittiRatioImageSize: bool = False
    # It will use a 192x640 size for input images.

    PyDNet2_usage: bool = False
    # It means that the model that will be used is PyDNet2 instead of PyDNet.

    def __init__(self):
        count = 0
        if ConfigCluster.HSV_processing is not None and ConfigCluster.HSV_processing:
            ConfigCluster.checkpoint_path += "HSV/"
            count += 1
        elif (
            ConfigCluster.BlackAndWhite_processing is not None
            and ConfigCluster.BlackAndWhite_processing
        ):
            ConfigCluster.checkpoint_path += "B&W/"
            count += 1
        elif (
            ConfigCluster.KittiRatioImageSize is not None
            and ConfigCluster.KittiRatioImageSize
        ):
            ConfigCluster.checkpoint_path += "192x640/"
            ConfigCluster.image_height = 192
            ConfigCluster.image_width = 640
            count += 1
        elif ConfigCluster.PyDNet2_usage is not None and ConfigCluster.PyDNet2_usage:
            ConfigCluster.checkpoint_path += "PyDNet2/"
            count += 1
        elif (
            ConfigCluster.VerticalFlipAugmentation is not None
            and ConfigCluster.VerticalFlipAugmentation
        ):
            ConfigCluster.checkpoint_path += "VFlip/"
            count += 1
        if count > 1:
            raise Exception(
                "Can't have more than one experimental configuration turned on!"
            )
