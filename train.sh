#!/bin/bash

#SBATCH --job-name TCSVT_exp_t
#SBATCH --error=error/train_error-%A-%x.txt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=22G
#SBATCH --time=72:00:00
#SBATCH --partition batch_grad
#SBATCH -o slurm/logs/slurm-%A-%x.out

pip install easydict
pip install Pillow==8.3.0
pip install opencv-contrib-python
pip install numpy==1.19.5

cd src

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py

exit 0
