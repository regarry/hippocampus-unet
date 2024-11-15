#!/bin/bash
#BSUB -n 1
#BSUB -W 2:00
#BSUB -q gpu
#BSUB -R "select[l40s]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"

#BSUB -o ./logs/%J.out
#BSUB -e ./logs/%J.out

source ~/.bashrc
source /usr/share/Modules/init/bash
module load cuda/11.0
conda activate /usr/local/usrapps/lsmsmart/fyu7/unet # enter the full path to the conda environment
#conda activate /rsstu/users/t/tghashg/MADMbrains/Ryan/conda/unet 
hostname
nvidia-smi
nvcc --version

python training1113.py --validation 20 --batch_size 10 --epochs 50 --scale .5 --classes 3