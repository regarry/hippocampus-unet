#!/bin/bash
#BSUB -n 1
#BSUB -W 24:00
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -o ./logs/%J.out
#BSUB -e ./logs/%J.out
source ~/.bashrc
module load cuda/11.0
conda activate /usr/local/usrapps/$GROUP/$USER/unet # enter the full path to the conda environment
#conda activate /rsstu/users/t/tghashg/MADMbrains/Ryan/conda/unet 
hostname
nvidia-smi
nvcc --version

# Get the current date and time
datetime=$(date '+%Y%m%d_%H%M%S')

# Create the output directory
output_dir="./data/testset/masks_$datetime"
mkdir -p $output_dir

python predict.py -s 0.5 -m ./checkpoints/checkpoint_epoch10.pth -i ./data/testset/imgs -o $output_dir