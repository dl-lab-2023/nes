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
from own_pipeline.containers.baselearner import load_baselearner, Baselearner, model_seeds
from own_pipeline.train_baselearners_rs import get_search_mode_appendix
from own_pipeline.util import enable_logging


def parse_arguments() -> Namespace:
    logging.info("parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openml_task_id",
        type=int,
    )
    parser.add_argument("--ensemble_size",
                        type=int)
    parser.add_argument(
        "--search_mode",
        type=str,
        required=True,
        choices=['hp', 'nas', 'initweights']
    )

    return parser.parse_args()


def load_baselearners(args: Namespace) -> Tuple[set[int], List[Baselearner]]:
    logging.info("loading baselearners...")

    baselearner_dir = f"./saved_model/task_{args.openml_task_id}{get_search_mode_appendix(args)}"
    ensemble_dir = f"./saved_ensembles/task_{args.openml_task_id}{get_search_mode_appendix(args)}"

    id_set: set[int] = torch.load(f"{ensemble_dir}/ensemble_{args.ensemble_size}_baselearners.pt")
    POOL_NAME = "own_rs"

    model_seed_list = [model_seeds(arch=seed, init=seed, scheme=POOL_NAME) for seed in id_set]
    baselearners = [
        load_baselearner(
            model_id=k,
            load_nn_module=False,
            baselearner_dir=baselearner_dir
        )
        for k in model_seed_list
    ]
    # move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for b in baselearners:
        b.to_device(device)
    return id_set, baselearners


def load_ensemble(args: Namespace, baselearners: List[Baselearner]) -> Ensemble:
    logging.info("loading ensemble...")
    return Ensemble(baselearners, bsl_weights=None) 


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
    ensemble_statistics_dir = f"./ensemble_stats/task_{args.openml_task_id}{get_search_mode_appendix(args)}"

    logging.info("saving...")
    Path(ensemble_statistics_dir).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(ensemble_statistics_dir, f"ensemble_{args.ensemble_size}_baselearners_performance.json"), 'w') as f:
        json.dump({
            "baselearners": list(baselearner_ids),  # set is not serializable
            "evaluation": ensemble.evals,
            "evaluation_avg_baselearner": dict(ensemble.avg_baselearner_evals),
            "evaluation_oracle": ensemble.oracle_evals
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
