# PyDNet-torch

Studying, implementing and experimenting with the PyTorch version of [PyDNet v1](https://github.com/mattpoggi/pydnet).

Everything was made by me (if not cited otherwise).

Used & studied models:

-   [PyDNet v1](https://github.com/mattpoggi/pydnet) (studied the original code and reproduced the experiment to verify paper results);
-   [monodepth](https://github.com/mrharicot/monodepth) (used under the hood to train PyDNet);
-   [micromind](https://github.com/micromind-toolkit/micromind/tree/dev) (used XiNet trying to make PyDNet v2 more efficient).

# General information

In this repository you can find these main folders:

-   `/PyDNet-torch`: my PyDNet v1 and v2 implementation made using the PyTorch framework;
-   `/studied_models`: this folder contains:
    -   The original PyDNet v1 implementation, written in Tensorflow 1.X (with the code of monodepth that was used for training and evalutation);
    -   A migration of the code of PyDNet v1 to Tensorflow 2.X (unfortunately it doesn't work on the GPU);
    -   The code of micromind.
-   `/variants_and_experimentations`: attempts to transform PyDNet v2 in a more efficient and performant CNN through the use of the XiNet architecture (watch in the `/variants_and_experimentations/PyXiNet` folder);
-   `/slurm_files`: the folder that contains slurm files that were used to execute the code inside of the training cluster;
-   `/10_test_images`: 10 different images from the KITTI dataset, that will be used in the evalutaion phase to compute the average inference time using only the CPU.

## Info

> Note: `wandb` was used to log the different losses. To use it you'll have to:
>
> -   [create an account](https://wandb.ai/login?signup=true);
> -   install the package locally;
> -   configure the packate with your account information.

# PyDNet-torch

## Requirements

```bash
# Create the conda environment (use your preferred name)
conda create -n <environmentName>
# Activate the conda environment
conda activate <environmentName>
# Install the required packages (I'll use conda for torch)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# Install the required packages (I'll use pip for everything else)
pip install wandb pandas matplotlib Pillow
```

> **WARNING**: if you want to use the `--use=webcam` flag, your system must have the `ffmpeg` command installed and know that this functionality was only tested on a macOS device with an M1 Pro ARM CPU. I had to use it because ARM chips can't use open-cv yet.

> **IMPORTANT**: choose the cuda version based on the cuda version of your system.

## Configurations

To make things smoother to try and test, this project is based on configurations, lowering the amount of cli parameters you have to care for while executing the scripts.

You can find two examples of configurations inside the `PyDNet-torch/Configs` folder and two other examples inside the `variants_and_experimentations/PyXiNet` folder. Every configuration parameter that's not obvious it's well documented in the provided examples.

You'll want to create your own configuration or modify the existing ones to specify different parameters, including the dataset path, the image resolution, and so on.

To create a custom configuration, copy one of the examples (i.e. `ConfigHomeLab.py`) and modify it to your likings.

After you created your own configuration, you have to:

-   Import it inside of `PyDNet-torch/Config.py`, and add the conditional logic to use your specified configuration;
-   Import it inside of `PyDNet-torch/testing.py` and add it as the possible types of the parameter `config` inside of the `evaluate_on_test_set` function;
-   In the `PyDNet-torch/main.py` file you could add to the helper of the parser of the `--env` parameter, the name that has to be provided in order to select your new configuration.

After that you are done!

## Training

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This will generate the checkpoint of the last epoch and will maintain the checkpoint that had the best performance on the test set, inside the directory specified by the `checkpoint_path` attribute of the selected configuration.

```bash
cd PyDNet-torch # To move into the model's folder
python3 main.py --mode=train --env=<NameOfTheConfigurationYouWantToUse>
```

## Testing

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This is used to generate the `disparities.npy` file. It will contain the disparities calculated for the images of the choosen test set.

The file will be placed inside the directory specified by the `output_directory` attribute of the selected configuration.
To execute the testing you should have a checkpoint first, specified by the `checkpoint_to_use_path` attribute of the selected configuration.

```bash
cd PyDNet-torch # To move into the model's folder
python3 main.py --mode=test --env=<NameOfTheConfigurationYouWantToUse>
```

## Evaluating

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This is used to evaluate the model (using the evaluation techniques utilized by PyDNet and Monodepth) on the `disparities.npy` file, generated from the test set (look at the testing section).

It will also measure the average of the inference time, of the model on 10 different images (that you can find inside of the `10_test_images/` folder), using only the CPU as the computing device.

To execute the evalutation you should have a checkpoint first, specified by the `checkpoint_to_use_path` attribute , and a `disparities.npy` file inside the folder specified by the `output_directory` attribute of the selected configuration.

```bash
cd PyDNet-torch # To move into the model's folder
python3 main.py --mode=eval --env=<NameOfTheConfigurationYouWantToUse>
```

## Using

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This will create a depth map image in the same folder of the image that was provided to the model.

To use the model on an image you should have a checkpoint first, specified by the `checkpoint_to_use_path` attribute of the selected configuration.

```bash
cd PyDNet-torch # To move into the model's folder
python3 main.py --mode=use --env=<NameOfTheConfigurationYouWantToUse> --img_path=<pathOfTheImageYouWantToUse>
```

# Requirements for the original PyDNet

> **WARNING**: this original implementation uses an outdated and deprecated version of Tensorflow, that was using really old versions of CUDA and cuDNN.
>
> Even if you can achieve to download the right version of the packages needed to make it work, you will have to re-configure the drivers in your machine to make it use the GPU.
>
> Another possible way could be to use nvidia-docker (never tried).

In case you want to use the original PyDNet Tensorflow 1.X implementation, these are the commands you can use to configure the python environment:

```bash
# Create the conda environment (use your preferred name)
conda create -n <environmentName> python=3.7
# Activate the conda environment
conda activate <environmentName>
# Install the required packages (I'll use pip)
pip install protobuf==3.20 tensorflow_gpu=1.13.2 scipy=1.2 matplotlib wandb
```

In case you want to use the migrated PyDNet Tensorflow 2.X implementation, these are the commands you can use to configure the python environment (it doesn't work on the GPU):

```bash
# Create the conda environment (use your preferred name)
conda create -n <environmentName>
# Activate the conda environment
conda activate <environmentName>
# Install the required packages (I'll use pip)
pip install tensorflow Pillow matplotlib wandb
```
