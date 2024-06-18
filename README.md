# Depth

Studying and implementing the Pytorch version of [PyDNet v1](https://github.com/mattpoggi/pydnet)

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
pip install wanb pandas opencv-python
```

> IMPORTANT: choose the cuda version based on the cuda version of your system.

### Training

You have to make a configuration first. In the `Configs` folder you can find examples of the configurations I used in my homelab and in the department cluster.
Once you created your own configuration, you have to add the selectable option inside the `Config.py` file.

Every single configuration parameter it's well documented inside the examples you can find in the `Configs` folder.

After you created the configuration you have to add it to the `main.py` file as a selectable option in the _arguments parameters_.

```bash
cd PyDNet-torch # To move into the model's folder
python3 main.py --mode=train --env=<NameOfTheConfigurationYouWantToUse>
```

### Evaluating

### Using

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
