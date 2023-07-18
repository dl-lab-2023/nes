#!/bin/bash
set -e

max_seed=10

while read taskid; do
  echo "training on task: $taskid"
  for ((i=0; i<=max_seed; i++)); do
    python -m own_pipeline.train_baselearners_rs --seed $i --openml_task_id $taskid
  done
done <own_pipeline/task_ids.txt


