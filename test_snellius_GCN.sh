#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH -o jupyter-notebook-py-GCN.out

module purge
module load 2022
# module load Anaconda3/2023.07-2
# module load CUDA/12.1.1
# module load cuDNN/8.9.2.26-CUDA-12.1.1
module load CUDA/11.7.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

# This loads the anaconda virtual environment with our packages
# source $HOME/.bashrc
# conda activate

# source activate GNN_gpu

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
