#!/bin/bash
#SBATCH --nodelist=dellcuda2
#SBATCH --job-name=PyXiNetM4
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --error=../variants_and_experimentations/PyXiNet/outputfiles/outputs/PyXiNetM4.err
#SBATCH --output=../variants_and_experimentations/PyXiNet/outputfiles/outputs/PyXiNetM4.out


source /home/rtoniolo/.bashrc
conda activate PyDNet-torch

echo "Executing python code"
# execution of training code
python3 /home/rtoniolo/Depth/variants_and_experimentations/PyXiNet/main.py --mode=train --env=Cluster