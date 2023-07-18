#!/bin/bash
set -e

max_seed=10

for ((i=0; i<=max_seed; i++)); do
   python -m own_pipeline.HPO_random_search --seed $i --openml_task_id=233088
done
