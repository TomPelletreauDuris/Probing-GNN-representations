#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=TestJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:05:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

# This loads the anaconda virtual environment with our packages
# source $HOME/.bashrc
# conda activate

source activate GNN_gpu

srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#print the GPU driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader

#print the python version
python --version

# Run the actual experiment. 
python test.py
python <<EOF
import torch
print(torch.cuda.is_available())
EOF
