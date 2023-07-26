# Base Learners vs. Ensembles for Tabular Datasets: Comparing HPO, NAS, and different Weight Initializations with Random Searc

## Introduction
This repo contains the code accompanying the paper:
[Neural Ensemble Search for Uncertainty Estimation and Dataset Shift](https://arxiv.org/abs/2006.08573)
(Authors: Sheheryar Zaidi*, Arber Zela*, Thomas Elsken, Chris Holmes, Frank Hutter and Yee Whye Teh.)

The project is used to extend and try these ensembling techniques on tabular dataset that are used in this paper:
[Well-tuned Simple Nets Excel on Tabular Datasets](https://arxiv.org/pdf/2106.11189)
(Authors: Arlind Kadra, Marius Lindauer, Frank Hutter, Josif Grabocka)

## Setting up virtual environment

First, clone and cd to the root of repo:

```
git clone https://github.com/dl-lab-2023/nes.git
cd nes
```

We used Python 3.9 and please check `requirements.txt` for our torch version and other requirements for running our experiments. For reproducibility, we recommend using these python and CUDA versions.

## Running the experiments

The structure for running the random Neural Ensemble Search consists of three steps: train the base learners, apply ensemble selection and evaluate the final ensembles. We perform these steps 3 times, we first train on random Hyperparameters but fixed architecture, then we train on the best hyperparameter config from the previous step on a random NAS search method, after which we apply this method on the best architecture and hyperparameter configuration obtained from the previous steps with different weight initialization. We ensemble each result and we visualize the results with plotting functions.

Instructions for running on a cluster with the MOAB cluster manager, e.g., Nemo:

### Step 1: NES-RS for HPO

1. Train base learners:

    `$ msub -t 0-9249 -v NUM_SEEDS_PER_TASK=250 -l walltime=4:00:00 ../own_pipeline/cluster_train_baselearners_rs_hp.sh`

2. Create ensemble:

    `$ msub -t 0-36 -v MAX_SEED=250 -v ENSEMBLE_SIZE=20 -l walltime=4:00:00 ../own_pipeline/cluster_create_ensemble_hp.sh`

3. Evaluate ensemble:

    `$ msub -t 0-36 -v ENSEMBLE_SIZE=20 -l walltime=4:00:00 ../own_pipeline/cluster_eval_ensemble_hp.sh`

### Step 2: NES-RS for NAS

1. Train base learners:

    `$ msub -t 0-9249 -v NUM_SEEDS_PER_TASK=250 -l walltime=4:00:00 ../own_pipeline/cluster_train_baselearners_rs_nas.sh`

2. Create ensemble:

    `$ msub -t 0-36 -v MAX_SEED=250 -v ENSEMBLE_SIZE=20 -l walltime=4:00:00 ../own_pipeline/cluster_create_ensemble_nas.sh`

3. Evaluate ensemble:

    `$ msub -t 0-36 -v ENSEMBLE_SIZE=20 -l walltime=4:00:00 ../own_pipeline/cluster_eval_ensemble_nas.sh`

### Step 3: NES-RS for different model weight initializations (MWIs)

1. Train base learners:

    `$ msub -t 0-9249 -v NUM_SEEDS_PER_TASK=250 -l walltime=4:00:00 ../own_pipeline/cluster_train_baselearners_rs_initweights.sh`

2. Create ensemble:

    `$ msub -t 0-36 -v MAX_SEED=250 -v ENSEMBLE_SIZE=20 -l walltime=4:00:00 ../own_pipeline/cluster_create_ensemble_initweights.sh`

3. Evaluate ensemble:

    `$ msub -t 0-36 -v ENSEMBLE_SIZE=20 -l walltime=4:00:00 ../own_pipeline/cluster_eval_ensemble_initweights.sh`

### Step 4: Analyze ensemble accuracy based on the number of base learners considered for ensemble building

For each dataset (=task id), run the following command. Please replace the task id `233091` in this example with the task id that should be evaluated.

`$ msub -t 0-12 -v TASK_ID=233091 -l walltime=4:00:00 ../../own_pipeline/cluster_create_multi_ensemble.sh`

### Step 5: Generate plots

For each dataset (=task id), run the following command. Please replace the task id `233091` in this example with the task id that should be evaluated.

`$ python -m own_pipeline.plotting.main --ensemble_stats_path ensemble_stats/ --saved_model_path saved_model/ --save_path plots/ --load_cluster_json_path ensemble_stats/ensemble_stats.json --multi_ensemble_stats_dir multi_ensemble_stats/ --taskid 233091`
