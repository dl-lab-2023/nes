"""reference: nes/ensemble_selection/evaluate_ensembles.py"""

import argparse
import os
from argparse import Namespace
from typing import List

import torch

from own_pipeline.containers.ensemble import Ensemble
from nes.ensemble_selection.utils import args_to_device
from own_pipeline.containers.baselearner import load_baselearner, Baselearner, model_seeds


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Index of GPU device to use. For CPU, set to -1. Default: -1.",
    )
    parser.add_argument(
        "--baselearner_dir",
        type=str,
        default="saved_model"
    )
    parser.add_argument(
        "--ensemble_dir",
        type=str,
        default="saved_ensembles"
    )
    parser.add_argument(
        "--ensemble_name",
        type=str,
        help="create an ensemble with 'create_ensemble' and pass the name of the .pt file here "
             "- do not include the '.pt' ending",
        required=True
    )

    return parser.parse_args()


def load_baselearners(args: Namespace) -> List[Baselearner]:
    id_set: set[int] = torch.load(f"{args.ensemble_dir}/{args.ensemble_name}.pt")
    POOL_NAME = "own_rs"

    model_seed_list = [model_seeds(arch=seed, init=seed, scheme=POOL_NAME) for seed in id_set]
    baselearners = [
        load_baselearner(
            model_id=k,
            load_nn_module=False,
            baselearner_dir=args.baselearner_dir
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
    print(dict(ensemble.avg_baselearner_evals))


if __name__ == '__main__':
    main()
