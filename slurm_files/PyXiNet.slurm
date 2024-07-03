#!/bin/bash
#SBATCH --nodelist=debug01
#SBATCH --job-name=PyXiNet
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --error=../PyXiNet/outputfiles/outputs/PyXiNet.err
#SBATCH --output=../PyXiNet/outputfiles/outputs/PyXiNet.out


source /home/rtoniolo/.bashrc
conda activate PyDNet-torch

echo "Executing python code"
# execution of training code
python3 /home/rtoniolo/Depth/PyXiNet/main.py --mode=train --env=Cluster --efficiency=L