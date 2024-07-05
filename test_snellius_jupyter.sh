
#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -p thin
#SBATCH -o jupyter-notebook-job.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load JupyterHub/4.0.2-GCCcore-12.3.0

source activate GNN_gpu

# Choose random port and print instructions to connect
PORT=`shuf -i 5000-5999 -n 1`
LOGIN_HOST=${SLURM_SUBMIT_HOST}-pub.snellius.surf.nl
BATCH_HOST=$(hostname)

echo "To connect to the notebook type the following command into your local terminal:"
echo "ssh -N -J ${USER}@${LOGIN_HOST} ${USER}@${BATCH_HOST} -L ${PORT}:localhost:${PORT}"
echo
echo "After connection is established in your local browser go to the address:"
echo "http://localhost:${PORT}"
 
jupyter notebook --no-browser --port $PORT
