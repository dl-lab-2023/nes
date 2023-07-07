#!/bin/bash
#SBATCH -a 0-399
#SBATCH -o ./cluster_logs/own_pipeline/%A-%a.o
#SBATCH -e ./cluster_logs/own_pipeline/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J own_pipeline # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate virtual environment
source .venv/bin/activate

# Arrayjob
PYTHONPATH=$PWD python own_pipeline/HPO_random_search.py --seed $SLURM_ARRAY_TASK_ID

# Done
echo "DONE"
echo "Finished at $(date)"
