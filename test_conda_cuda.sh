#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH -o Output_conda_cuda.out
#SBATCH --cpus-per-task=18

module purge
module load 2022
module load CUDA/11.7.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

# module load Anaconda3/2023.07-2
# module load Anaconda3/2022.05
# # Activate the Conda environment
# source $HOME/.bashrc
# source activate GNN_gpu
# conda activate GNN_gpu

# # Ensure Conda environment includes matching CUDA and cuDNN
# conda install -c conda-forge cudatoolkit=11.7 cudnn=8.2
# conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch

# # Run the Python script to check CUDA availability
srun python -uc "import torch; print('CUDA version:', torch.version.cuda); print('cuDNN version:', torch.backends.cudnn.version()); print('GPU available?', torch.cuda.is_available())"

# Print the GPU driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
