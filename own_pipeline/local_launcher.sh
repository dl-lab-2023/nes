#!/bin/bash
set -e

for ((i=0; i < 13; i++)); do
  echo Starting with JOBARRAYINDEX=$i
  TASK_ID=233096 MOAB_JOBARRAYINDEX=$i ./own_pipeline/cluster_create_multi_ensemble.sh
done



# max_seed=10

# while read taskid; do
#   echo "now processing taskid: $taskid"
#   #for ((i=0; i<=max_seed; i++)); do
#   #python -m own_pipeline.train_baselearners_rs --openml_task_id $taskid --seed $i --search_mode hp
#   #python -m own_pipeline.create_ensemble --max_seed $max_seed --ensemble_size 5 --openml_task_id $taskid --search_mode hp
#   python -m own_pipeline.evaluate_ensemble --openml_task_id $taskid --ensemble_size 20 --search_mode hp
#   #done
# done <own_pipeline/task_ids.txt


