#!/bin/bash
#SBATCH --job-name=pydnet_torch
#SBATCH --error=../PyDNet-torch/outputs/pydnet_torch.err
#SBATCH --output=../PyDNet-torch/outputs/pydnet_torch.out
#SBATCH --partition=testing
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --nodelist=vgpu6-0
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00


source /home/rtoniolo/.bashrc
conda activate PyDNet-torch

echo "Executing python code"
# execution of training code
python3 /home/rtoniolo/Depth/PyDNet-torch/main.py --mode=train --env=Cluster