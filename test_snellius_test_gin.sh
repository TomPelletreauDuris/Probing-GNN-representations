#!/bin/bash

#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH -o notebook_ASD_gin.out

module purge
module load 2022
module load CUDA/11.7.0
module load IPython/8.5.0-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
# module load 2023 
# module load IPython/8.14.0-GCCcore-12.3.0
# module load PyTorch/1.12.0-foss-2023a-CUDA-11.7.0


srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#print the GPU driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader

#print the python version
python --version

# Run the actual experiment. 
python test_gin.py
# python <<EOF
# import torch
# print(torch.cuda.is_available())
# EOF
