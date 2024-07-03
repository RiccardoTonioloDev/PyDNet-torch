#!/bin/bash
#SBATCH --nodelist=dellcuda2
#SBATCH --job-name=pydnet_torch_CS
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --error=../PyDNet-torch/outputfiles/outputs/pydnet_torch_CS.err
#SBATCH --output=../PyDNet-torch/outputfiles/outputs/pydnet_torch_CS.out


source /home/rtoniolo/.bashrc
conda activate PyDNet-torch

echo "Executing python code"
# execution of training code
python3 /home/rtoniolo/Depth/PyDNet-torch/main.py --mode=train --env=Cluster
