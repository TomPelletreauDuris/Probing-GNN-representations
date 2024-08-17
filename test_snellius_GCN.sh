#!/bin/bash

#SBATCH -t 50:00:00
#SBATCH -N 1
#SBATCH -p rome
#SBATCH --cpus-per-task=192
#SBATCH --mem=336G
#SBATCH -o jupyter-notebook-py-GCN.out

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