#!/bin/bash
# See https://wiki.bwhpc.de/e/NEMO/Moab#Job_array_example

# For this job, you need to set the $NUM_SEEDS_PER_TASK environment variable!
# Example command for submitting:
# msub -t 0-99 -v NUM_SEEDS_PER_TASK=10 -l nodes=1:ppn=1 own_pipeline/cluster_train_baselearners_rs.sh

handle_error() {
    echo "ERROR processing the task"
}

set -eE
trap 'handle_error' ERR

# The NEMO cluster offers getting conda using the "module" feature
# Unfortunately, the the most recent available Python version was 3.6.
# Therefore we installed Miniconda manually using https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links

MOAB_WORKSPACE_NAME=dl-lab
CONDA_BIN=$(ws_find $MOAB_WORKSPACE_NAME)/conda/bin/conda
CONDA_WORKSPACE_NAME=dl-lab

echo "Started at $(date)"
echo "MOAB_JOBARRAYINDEX (this job id): $MOAB_JOBARRAYINDEX"

cd $(ws_find $MOAB_WORKSPACE_NAME)/nes
#echo "PWD: $PWD"

eval "$($CONDA_BIN shell.bash hook)"
conda activate $CONDA_WORKSPACE_NAME
#echo Activated conda environment

#echo Running work...
TASK_NUM=$((MOAB_JOBARRAYINDEX / NUM_SEEDS_PER_TASK))
TASK_NUM=$((TASK_NUM + 1)) # Start counting at 1 instead of at 0
TASK_ID=$(sed "${TASK_NUM}q;d" own_pipeline/task_ids.txt) # Get line with number $TASK_NUM from the text file
SEED=$((MOAB_JOBARRAYINDEX % NUM_SEEDS_PER_TASK))

echo NUM_SEEDS_PER_TASK=$NUM_SEEDS_PER_TASK
echo TASK_NUM=$TASK_NUM
echo TASK_ID=$TASK_ID
echo SEED=$SEED
python -m own_pipeline.train_baselearners_rs --openml_task_id=$TASK_ID --seed $SEED --search_mode nas

echo "Finished at $(date)"
