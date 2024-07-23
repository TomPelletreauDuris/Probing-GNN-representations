#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -o Output_conda_cuda.out
#SBATCH --cpus-per-task=18

module purge
module load 2023

# Activate the Conda environment
source $HOME/.bashrc
conda activate GNN_gpu

# Ensure Conda environment includes matching CUDA and cuDNN
conda install -c conda-forge cudatoolkit=11.7 cudnn=8.2
conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch

# Run the Python script to check CUDA availability
srun python -uc "import torch; print('CUDA version:', torch.version.cuda); print('cuDNN version:', torch.backends.cudnn.version()); print('GPU available?', torch.cuda.is_available())"

# Print the GPU driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
