# PyDNet-torch

Studying and implementing the PyTorch version of [PyDNet v1](https://github.com/mattpoggi/pydnet)

Used & studied models:

-   [PyDNet v1](https://github.com/mattpoggi/pydnet)
-   [monodepth](https://github.com/mrharicot/monodepth) (used under the hood to train PyDNet)

# General information

In this repository you can find four main folders:

-   `/outputfiles`: the folder that contains the checkpoints of the models;
-   `/slurm_files`: the folder that contains slurm files that were used to execute the code inside of the cluster.
-   `/PyDNet-tf1`: the original PyDNet v1 in its tensorflow 1.X implementation;
-   `/PyDNet-tf2`: a PyDNet v1 implementation made migrating the code to tensorflow 2.X (unfortunately it doesn't seem to work with the GPU);
-   `/PyDNet-torch`: a PyDNet v1 implementation made using the Pytorch framework.

## Info

> Note: `wandb` was used to log the different losses. To use it you'll have to [create an account](https://wandb.ai/login?signup=true) and then configure the plugin to make it work.

# Models

## PyDNet-torch

### Requirements

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

> **WARNING**: if you want to use the `--use=webcam` flag, your system must have the `ffmpeg` command installed and know that this functionality was only tested on a macOS device with an ARM CPU. I use it because ARM chips can't use open-cv yet.

> IMPORTANT: choose the cuda version based on the cuda version of your system.

### Configurations

To make things smoother to try and test, this projects is based on configurations, lowering the amount of cli parameters you have to care for while executing the scripts.

You can find two examples of configurations inside the `Configs` folder. Every configuration parameter that's not obvious it's well documented in the provided examples.

You'll want to create you own configuration to specify different parameters, including the dataset path, the image resolution, and so on.

To create a custom configuration, create one copying one of the examples and modify it to your likings.

After you created your own configuration, you have to:

-   Modify the `testing.py` so that the string that represents your configuration is a part of the type literal that represent the type of the `env` parameter;
-   Modify the `evaluating.py` so that the string that represents your configuration is a part of the type literal that represent the type of the `env` parameter;
-   Modify the `using.py` so that the string that represents your configuration is a part of the type literal that represent the type of the `env` parameter.

### Training

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This will generate the checkpoint of the last epoch and will maintain the checkpoint that had the best performance on the test set.

Those will be found inside the `outputfiles/checkpoints` folder.

```bash
cd PyDNet-torch # To move into the model's folder
python3 main.py --mode=train --env=<NameOfTheConfigurationYouWantToUse>
```

### Testing

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This is used to generate the `disparities.npy` file. It will contain the disparities calculated for the images inside the provided test set.

```bash
cd PyDNet-torch # To move into the model's folder
python3 main.py --mode=test --env=<NameOfTheConfigurationYouWantToUse>
```

### Evaluating

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This is used to evaluate the model (using the evaluation techniques utilized by PyDNet and Monodepth) on the `disparities.npy` file, generated from the test set (look at the testing section).

```bash
cd PyDNet-torch # To move into the model's folder
python3 main.py --mode=eval --env=<NameOfTheConfigurationYouWantToUse>
```

### Using

**IMPORTANT**: make sure that the program it's using the right configuration as explained in the configurations section.

This will create a depth map image in the same folder of the image that was provided to the model.

```bash
cd PyDNet-torch # To move into the model's folder
python3 main.py --mode=use --env=<NameOfTheConfigurationYouWantToUse> --img_path=<pathOfTheImageYouWantToUse>
```

## PyDNet-tf1

### Requirements

```bash
# Create the conda environment (use your preferred name)
conda create -n <environmentName> python=3.7
# Activate the conda environment
conda activate <environmentName>
# Install the required packages (I'll use pip)
pip install protobuf==3.20 tensorflow_gpu=1.13.2 scipy=1.2 matplotlib wandb
```

### Training

### Evaluating

### Using

## PyDNet-tf2

### Requirements

```bash
# Create the conda environment (use your preferred name)
conda create -n <environmentName>
# Activate the conda environment
conda activate <environmentName>
# Install the required packages (I'll use pip)
pip install tensorflow scipy matplotlib wandb
```

### Training

### Evaluating

### Using
