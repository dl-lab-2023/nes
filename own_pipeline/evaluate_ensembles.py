"""reference: nes/ensemble_selection/evaluate_ensembles.py"""

import argparse
import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

import torch

from own_pipeline.containers.ensemble import Ensemble
from nes.ensemble_selection.utils import args_to_device
from own_pipeline.containers.baselearner import load_baselearner, Baselearner, model_seeds
from own_pipeline.util import enable_logging


def parse_arguments() -> Namespace:
    logging.info("parsing arguments...")
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
        "--ensemble_statistics_dir",
        type=str,
        default="ensemble_statistics",
        help="directory to store all information for plotting"
    )
    parser.add_argument(
        "--ensemble_name",
        type=str,
        help="create an ensemble with 'create_ensemble' and pass the name of the .pt file here "
             "- do not include the '.pt' ending",
        required=True
    )

    return parser.parse_args()


def load_baselearners(args: Namespace) -> Tuple[set[int], List[Baselearner]]:
    logging.info("loading baselearners...")
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
    return id_set, baselearners


def load_ensemble(args: Namespace, baselearners: List[Baselearner]) -> Ensemble:
    logging.info("loading ensemble...")
    return Ensemble(baselearners, bsl_weights=None)  # TODO bsl_weights?


def evaluate_ensemble(ensemble: Ensemble):
    logging.info("evaluating ensemble...")

    ensemble.compute_preds()
    ensemble.compute_evals()
    ensemble.preds = None  # clear memory

    ensemble.compute_avg_baselearner_evals()

    ensemble.compute_oracle_preds()
    ensemble.compute_oracle_evals()
    ensemble.oracle_preds = None  # clear memory

    # ensemble.compute_disagreement()
    torch.cuda.empty_cache()

    logging.info(f"avg_baselearner_evals: {dict(ensemble.avg_baselearner_evals)}")
    logging.info(f"evals: {dict(ensemble.evals)}")


def save_data(args: Namespace, ensemble: Ensemble, baselearner_ids: set[int]):
    logging.info("saving...")
    Path(args.ensemble_statistics_dir).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(args.ensemble_statistics_dir, f"{args.ensemble_name}_performance.json"), 'w') as f:
        json.dump({
            "baselearners": list(baselearner_ids),  # set is not serializable
            "evaluation": ensemble.evals,
            "evaluation_avg_baselearner": dict(ensemble.avg_baselearner_evals),
        }, f, sort_keys=True, indent=4)


def main():
    args = parse_arguments()
    ids, baselearners = load_baselearners(args)
    ensemble = load_ensemble(args, baselearners)
    evaluate_ensemble(ensemble)
    save_data(args, ensemble, ids)


if __name__ == '__main__':
    enable_logging()
    main()
