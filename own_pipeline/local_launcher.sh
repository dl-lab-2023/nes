#!/bin/bash
set -e

max_seed=25

for ((i=0; i<=max_seed; i++)); do
   python own_pipeline/HPO_random_search.py --seed $i
done
