#!/bin/bash
# See https://wiki.bwhpc.de/e/NEMO/Moab#Job_array_example

set -e

# The NEMO cluster offers getting conda using the "module" feature
# Unfortunately, the the most recent available Python version was 3.6.
# Therefore we installed Miniconda manually using https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links

MOAB_WORKSPACE_NAME=dl-lab
CONDA_BIN=$(ws_find $MOAB_WORKSPACE_NAME)/conda/bin/conda
CONDA_WORKSPACE_NAME=dl-lab

echo "Started at $(date)"
echo "MOAB_JOBARRAYINDEX (this job id): $MOAB_JOBARRAYINDEX"
echo "MOAB_JOBARRAYRANGE (num total jobs of this task): $MOAB_JOBARRAYRANGE"

cd $(ws_find $MOAB_WORKSPACE_NAME)/nes
echo "PWD: $PWD"

eval "$($CONDA_BIN shell.bash hook)"
conda activate $CONDA_WORKSPACE_NAME
echo Activated conda environment

echo Running work...
python -m own_pipeline.HPO_random_search --seed $MOAB_JOBARRAYINDEX

echo "DONE"
echo "Finished at $(date)"
