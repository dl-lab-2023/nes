"""reference: nes/ensemble_selection/evaluate_ensembles.py"""

import argparse
import os
from argparse import Namespace
from typing import List

import torch

from own_pipeline.containers.ensemble import Ensemble
from nes.ensemble_selection.utils import args_to_device
from own_pipeline.baselearner import load_baselearner, Baselearner, model_seeds


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Index of GPU device to use. For CPU, set to -1. Default: -1.",
    )
    parser.add_argument(
        "--arch_id",
        type=int,
        default=None,
        help="Only used for DeepEns (RS) + ESA",
    )
    parser.add_argument(
        "--esa",
        type=str,
        default="beam_search",
        help="Ensemble selection algorithm. See nes/ensemble_selection/esas.py. Default: beam_search.",
    )
    parser.add_argument("--M", type=int, default=5, help="Ensemble size. Default: 5.")
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save ensemble evaluation data for eventual plotting.",
    )
    parser.add_argument(
        "--load_bsls_dir",
        type=str,
        help="Directory where the baselearners in the pool are saved. Will usually depend on --method.",
    )
    parser.add_argument(
        "--load_ens_chosen_dir",
        type=str,
        help="Directory where output of ensembles_from_pools.py is saved. *Only used when --method is nes_rs or nes_re.*",
    )
    parser.add_argument(
        "--incumbents_dir",
        type=str,
        help="Directory where output of rs_incumbents.py is saved.",
    )
    parser.add_argument(
        "--nes_rs_bsls_dir",
        type=str,
        help="Directory where nes_rs baselearners are saved. *Only used when --method is deepens_rs.*",
    )
    parser.add_argument(
        "--method",
        choices=["nes_rs", "nes_re", "deepens_rs", "deepens_darts",
                 "deepens_amoebanet_50k", "nes_rs_oneshot", "nes_re_50k",
                 "deepens_darts_anchor", "deepens_darts_50k", "nes_rs_50k",
                 "deepens_amoebanet", "deepens_gdas", "deepens_minimum",
                 "deepens_pcdarts", "darts_esa", "amoebanet_esa", "nes_rs_esa",
                 "darts_rs", "darts_hyper", "joint", "nes_rs_darts"],
        type=str,
    )
    parser.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "fmnist", "imagenet", "tiny"], type=str, help="Dataset."
    )

    parser.add_argument(
        "--validation_size",
        type=int,
        default=-1,
    )

    return parser.parse_args()


def load_baselearners(args: Namespace) -> List[Baselearner]:
    id_set: set[int] = torch.load("saved_ensembles/ensemble_5_baselearners.pt")
    POOL_NAME = "own_rs"
    BASELEARNER_DIR = "saved_model"

    model_seed_list = [model_seeds(arch=seed, init=seed, scheme=POOL_NAME) for seed in id_set]
    baselearners = [
        load_baselearner(
            model_id=k,
            load_nn_module=False,
            baselearner_dir=BASELEARNER_DIR
        )
        for k in model_seed_list
    ]
    # move to device
    for b in baselearners:
        b.to_device(args_to_device(args.device))
    return baselearners


def load_ensemble(args: Namespace, baselearners: List[Baselearner]) -> Ensemble:
    return Ensemble(baselearners, bsl_weights=None)  # TODO bsl_weights?


def evaluate_ensemble(ensemble: Ensemble):
    ensemble.compute_preds()
    ensemble.compute_evals()
    ensemble.preds = None  # clear memory

    ensemble.compute_avg_baselearner_evals()

    ensemble.compute_oracle_preds()
    ensemble.compute_oracle_evals()
    ensemble.oracle_preds = None  # clear memory

    # ensemble.compute_disagreement()
    torch.cuda.empty_cache()


def main():
    args = parse_arguments()
    baselearners = load_baselearners(args)
    ensemble = load_ensemble(args, baselearners)
    evaluate_ensemble(ensemble)
    print(ensemble.evals)


if __name__ == '__main__':
    main()
