#!/bin/bash
#SBATCH --nodelist=dellsrv1
#SBATCH --job-name=pydnet_torch
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --error=../PyDNet-torch/outputs/pydnet_torch.err
#SBATCH --output=../PyDNet-torch/outputs/pydnet_torch.out


source /home/rtoniolo/.bashrc
conda activate PyDNet-torch

echo "Executing python code"
# execution of training code
python3 /home/rtoniolo/Depth/PyDNet-torch/main.py --mode=train --env=Cluster