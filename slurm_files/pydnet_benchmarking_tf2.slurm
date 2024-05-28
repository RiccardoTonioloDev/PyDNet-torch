#!/bin/bash
#SBATCH --job-name=pydnet_benchmark
#SBATCH --error=../outputfiles/pydnet_benchmarking_tf_v1/pydnet.err
#SBATCH --output=../outputfiles/pydnet_benchmarking_tf_v1/pydnet.out
#SBATCH --partition=allgroups
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --nodelist=dellsrv1
#SBATCH --gres=gpu:V100-FHHL-16GB:1
#SBATCH --time=3-00:00:00


source /home/rtoniolo/.bashrc
conda activate PyDNet-tf2

echo "Executing python code"
# execution of training code
python3 /home/rtoniolo/Depth/PyDNet-tf2/monodepth_main.py \
  --mode train \
  --model_name PyDNet_tf2 \
  --data_path /home/rtoniolo/Datasets/kitti/ \
  --filenames_file /home/rtoniolo/Depth/PyDNet-tf2/utils/filenames/kitti_train_files_png.txt \
  --log_directory /home/rtoniolo/Depth/outputfiles/pydnet_benchmarking_tf_v1/tmp/