#!/bin/bash

#SBATCH --job-name pedestrian
#SBATCH --error=../error/train_error.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time=24:00:00
#SBATCH --partition batch_ugrad
#SBATCH -o ../slurm/slurm-%A-%x.out

pip install easydict
pip install Pillow==8.3.0

cd src

echo "\nvisualization\n"

# KAIST
python inference.py --FDZ original --model-path ../weights/10per_ours_cont.pth.tar000 --result-dir ../viz/KAIST/10per/ours --vis