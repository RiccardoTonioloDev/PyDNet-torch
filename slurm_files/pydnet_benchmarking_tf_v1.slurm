#!/bin/bash
#SBATCH --job-name=pydnet_benchmark
#SBATCH --error=../outputfiles/pydnet_benchmarking_tf_v1/pydnet.err
#SBATCH --output=../outputfiles/pydnet_benchmarking_tf_v1/pydnet.out
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=dellcuda1
#SBATCH --gres=gpu:A100-PCIE-40GB:1
#SBATCH --time=2-12:00:00

echo "Activating Conda"
# conda environment activation
conda activate pydnet_v1

echo "Executing python code"
# execution of training code
python3 /home/rtoniolo/Depth/pydnetV1/training_code/monodepth_main.py \
  --mode train \
  --model_name pydnet_v1_tf \
  --data_path /home/rtoniolo/Datasets/kitti \
  --filenames_file /home/rtoniolo/Depth/pydnetV1/training_code/utils/filenames/kitti_train_files.txt \
  --log_directory /home/rtoniolo/Depth/outputfiles/pydnet_benchmarking_tf_v1/tmp/