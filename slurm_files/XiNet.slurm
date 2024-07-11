#!/bin/bash
#SBATCH --nodelist=dellcuda1
#SBATCH --job-name=XiNet
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --error=../variants_and_experimentations/micromind/outputfiles/XiNet.err
#SBATCH --output=../variants_and_experimentations/micromind/outputfiles/XiNet.out


source /home/rtoniolo/.bashrc
conda activate micromind

echo "Executing python code"
# execution of training code
python3 /home/rtoniolo/Depth/variants_and_experimentations/micromind/recipes/image_classification/train.py cfg/xinet.py