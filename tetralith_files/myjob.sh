#!/bin/bash
#SBATCH -J 1D_ResNet152_training
#SBATCH -t 24:00:00
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.p.hohmann@gmail.com
#
# commands
./tetralith_script_1d_resnet152.py
