#!/bin/bash

#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -p genoa
## SBATCH --gpus-per-node=1
#SBATCH -o fc-gin-mdd-cuda.out


module purge
module load 2022
module load CUDA/11.7.0
# module load Anaconda3/2022.05
# module load Anaconda3/2023.07-2
#module load IPython/8.14.0-GCCcore-12.3.0
# module load JupyterHub/4.0.2-GCCcore-12.3.0
module load IPython/8.5.0-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
# module load JupyterHub/3.0.0-GCCcore-11.3.0

# source activate GNN_gpu

python3 FC_probing_GIN_MDD.py