#!/bin/bash
#SBATCH –-job-name=”pydnet_bncmk”
#SBATCH –-error=../outputfiles/pydnet_benchmarking_tf_v1/pydnet.err
#SBATCH –-output=../outputfiles/pydnet_benchmarking_tf_v1/hello.out
#SBATCH –-partition=allgroups
#SBATCH –-ntasks=1
#SBATCH –-mem=1G
#SBATCH --cpu-per-task=16
#SBATCH --nodelist=dellcuda1
#SBATCH --gres=gpu:A100-PCIE-40GB:1
#SBATCH –-time=2-12:00:00

# conda environment activation
conda activate pydnet_v1

# execution of training code
python ../pydnetV1/training_code/monodepth_main.py \
  --mode train \
  --model_name pydnet_v1_tf \
  --data_path ~/data/KITTI/ \
  --filenames_file ~/code/monodepth/utils/filenames/kitti_train_files.txt \
  --log_directory ../outputfiles/pydnet_benchmarking_tf_v1/tmp/