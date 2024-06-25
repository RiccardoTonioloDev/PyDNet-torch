#!/bin/bash
#SBATCH --nodelist=dellsrv3
#SBATCH --job-name=pydnet_torch_kittiratio
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --error=../PyDNet-torch/outputfiles/outputs/pydnet_torch_kittiratio.err
#SBATCH --output=../PyDNet-torch/outputfiles/outputs/pydnet_torch_kittiratio.out


source /home/rtoniolo/.bashrc
conda activate PyDNet-torch

echo "Executing python code"
# execution of training code
python3 /home/rtoniolo/Depth/PyDNet-torch/main.py --mode=train --env=Cluster