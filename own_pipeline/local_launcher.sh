#!/bin/bash
set -e

max_seed=10

#while read taskid; do
taskid=233091
echo "training on task: $taskid"
for ((i=0; i<=max_seed; i++)); do
  #python -m own_pipeline.train_baselearners_rs --openml_task_id $taskid --seed $i --search_mode hp
  #python -m own_pipeline.create_ensemble --max_seed $max_seed --ensemble_size 5 --openml_task_id $taskid --search_mode hp
  python -m own_pipeline.evaluate_ensemble --openml_task_id $taskid --ensemble_size 5 --search_mode hp
done
#done <own_pipeline/task_ids.txt


