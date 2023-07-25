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
ENSEMBLE_SIZE=20
MAX_SEED=$(( (MOAB_JOBARRAYINDEX + 1) * 20)) # MOAB_JOBARRAYINDEX starts at 0

echo "Starting HP..."
python -m own_pipeline.create_ensemble --max_seed $MAX_SEED --ensemble_size $ENSEMBLE_SIZE --openml_task_id $TASK_ID --out_dir saved_multi_ensembles --out_dir_subdir_suffix _$MAX_SEED --search_mode hp
echo "Evaluating..."
python -m own_pipeline.evaluate_ensemble --openml_task_id $TASK_ID --ensemble_size $ENSEMBLE_SIZE --ensembles_in_dir saved_multi_ensembles --ensembles_in_dir_subdir_suffix _$MAX_SEED --search_mode hp

echo "Starting NAS..."
python -m own_pipeline.create_ensemble --max_seed $MAX_SEED --ensemble_size $ENSEMBLE_SIZE --openml_task_id $TASK_ID --out_dir saved_multi_ensembles --out_dir_subdir_suffix _$MAX_SEED --search_mode nas
echo "Evaluating..."
python -m own_pipeline.evaluate_ensemble --openml_task_id $TASK_ID --ensemble_size $ENSEMBLE_SIZE --ensembles_in_dir saved_multi_ensembles --ensembles_in_dir_subdir_suffix _$MAX_SEED --search_mode nas

echo "Starting INITWEIGHTS..."
python -m own_pipeline.create_ensemble --max_seed $MAX_SEED --ensemble_size $ENSEMBLE_SIZE --openml_task_id $TASK_ID --out_dir saved_multi_ensembles --out_dir_subdir_suffix _$MAX_SEED --search_mode initweights
echo "Evaluating..."
python -m own_pipeline.evaluate_ensemble --openml_task_id $TASK_ID --ensemble_size $ENSEMBLE_SIZE --ensembles_in_dir saved_multi_ensembles --ensembles_in_dir_subdir_suffix _$MAX_SEED --search_mode initweights

echo "Finished at $(date)"
