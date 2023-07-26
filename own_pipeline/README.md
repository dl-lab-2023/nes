# Base Learners vs. Ensembles for Tabular Datasets: Comparing HPO, NAS, and different Weight Initializations with Random Searc

## Introduction
This repo contains the code accompanying the paper:

[Neural Ensemble Search for Uncertainty Estimation and Dataset Shift](https://arxiv.org/abs/2006.08573)

Authors: Sheheryar Zaidi*, Arber Zela*, Thomas Elsken, Chris Holmes, Frank Hutter and Yee Whye Teh.

The project is used to extend and try these ensembling techniques on tabular dataset that are used in this paper:

[Well-tuned Simple Nets Excel on Tabular Datasets](https://arxiv.org/pdf/2106.11189)

Authors: Arlind Kadra, Marius Lindauer, Frank Hutter, Josif Grabocka

## Setting up virtual environment

First, clone and cd to the root of repo:

```
git clone https://github.com/dl-lab-2023/nes.git
cd nes
```

We used Python 3.9 and please check `requirements.txt` for our torch version and other requirements for running our experiments. For reproducibility, we recommend using these python and CUDA versions.

## Running the experiments

The structure for running the random Neural Ensemble Search consists of three steps: train the base learners, apply ensemble selection and evaluate the final ensembles. We perform these steps 3 times, we first train on random Hyperparameters but fixed architecture, then we train on the best hyperparameter config from the previous step on a random NAS search method, after which we apply this method on the best architecture and hyperparameter configuration obtained from the previous steps with different weight initialization. We ensemble each result and we visualize with out plotting functions.

### Running NES

