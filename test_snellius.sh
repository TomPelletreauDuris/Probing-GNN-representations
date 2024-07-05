#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=TestJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output_%A.out


module purge
module load 2023
module load  Anaconda3/2023.07-2


#print the GPU driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader


## For Lisa and Snellius, modules are usually not needed
## https://userinfo.surfsara.nl/systems/shared/modules 

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate

#print the python version
python --version

# Run the actual experiment. 
python test.py
python <<EOF
import torch
print(torch.cuda.is_available())
EOF
