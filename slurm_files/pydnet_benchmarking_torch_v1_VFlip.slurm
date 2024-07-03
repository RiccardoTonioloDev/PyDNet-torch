#!/bin/bash
#SBATCH --nodelist=debug04
#SBATCH --job-name=pydnet_torch_VFlip
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --error=../PyDNet-torch/outputfiles/outputs/pydnet_torch_VFlip.err
#SBATCH --output=../PyDNet-torch/outputfiles/outputs/pydnet_torch_VFlip.out


source /home/rtoniolo/.bashrc
conda activate PyDNet-torch

echo "Executing python code"
# execution of training code
python3 /home/rtoniolo/Depth/PyDNet-torch/main.py --mode=train --env=Cluster
