#!/bin/bash
# See https://wiki.bwhpc.de/e/NEMO/Moab#Job_array_example

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
TASK_NUM=$((MOAB_JOBARRAYINDEX + 1)) # Start counting at 1 instead of at 0
TASK_ID=$(sed "${TASK_NUM}q;d" own_pipeline/task_ids.txt) # Get line with number $TASK_NUM from the text file
python -m own_pipeline.evaluate_ensemble --openml_task_id $TASK_ID --ensemble_size $ENSEMBLE_SIZE

echo "Finished at $(date)"
