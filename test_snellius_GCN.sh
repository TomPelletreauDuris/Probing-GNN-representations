#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -o jupyter-notebook-py-GCN.out
#SBATCH --cpus-per-task=18

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.1.1

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
python FC_probing_GCN.py
# python <<EOF
# import torch
# print(torch.cuda.is_available())
# EOF
