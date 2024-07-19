#!/bin/bash
#SBATCH --nodelist=dellsrv4
#SBATCH --job-name=PyXiNetB3
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --error=../variants_and_experimentations/PyXiNet/outputfiles/outputs/PyXiNetB3.err
#SBATCH --output=../variants_and_experimentations/PyXiNet/outputfiles/outputs/PyXiNetB3.out


source /home/rtoniolo/.bashrc
conda activate PyDNet-torch

echo "Executing python code"
# execution of training code
python3 /home/rtoniolo/Depth/variants_and_experimentations/PyXiNet/main.py --mode=train --env=Cluster