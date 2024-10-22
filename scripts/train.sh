#!/bin/bash
#BSUB -n 1
#BSUB -W 24:00
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -o ./logs/%J.out
#BSUB -e ./logs/%J.out
source ~/.bashrc
module load cuda/11.0
# conda activate /usr/local/usrapps/$GROUP/$USER/unet # enter the full path to the conda environment
conda activate /rsstu/users/t/tghashg/MADMbrains/Ryan/conda/unet 
hostname
nvidia-smi
nvcc --version

python train.py --validation 20 --batch_size 50 --epochs 100 --scale .5