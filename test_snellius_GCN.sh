#!/bin/bash

#SBATCH -t 30:00:00
#SBATCH -N 1
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH -o jupyter-notebook-py-GCN.out
#SBATCH --cpus-per-task=18

module purge
module load 2022
module load CUDA/11.7.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

source activate GNN_gpu

srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

# Print the GPU driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Print the Python version
python --version

# Run the actual experiment
python FC_probing_GCN.py