#!/bin/bash
#SBATCH --job-name=pydnet_benchmark
#SBATCH --error=../outputfiles/pydnet_benchmarking_tf_v1/pydnet.err
#SBATCH --output=../outputfiles/pydnet_benchmarking_tf_v1/pydnet.out
#SBATCH --partition=reserved0
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=dellcuda1
#SBATCH --gres=gpu:A100-PCIE-40GB:1
#SBATCH --time=2-12:00:00

# conda environment activation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pydnet_v1

# execution of training code
python ../pydnetV1/training_code/monodepth_main.py \
  --mode train \
  --model_name pydnet_v1_tf \
  --data_path ../../Datasets/kitti \
  --filenames_file ../pydnetV1/training_code/utils/filenames/kitti_train_files.txt \
  --log_directory ../outputfiles/pydnet_benchmarking_tf_v1/tmp/