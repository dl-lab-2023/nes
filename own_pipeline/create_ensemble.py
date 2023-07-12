import logging
import os
import dill as pickle
import torch
import argparse
from collections import defaultdict
from pathlib import Path

from nes.ensemble_selection.utils import (
    args_to_device,
)
from nes.ensemble_selection.esas import registry as esas_registry
from own_pipeline.containers.baselearner import load_baselearner, model_seeds
from own_pipeline.util import enable_logging


def run_esa(M, population, esa, val_severity, validation_size=-1, diversity_strength=None):
    model_ids_pool = list(population.keys())
    models_preds_pool = {
        x: population[x].preds for x in model_ids_pool
    }

    if validation_size > -1:
        assert validation_size > 0, "Validation size cannot be 0."
        _models_preds_pool = {}
        for x, tensor_data in models_preds_pool.items():
            preds, labels = tensor_data.tensors
            assert (validation_size <= len(preds)), "Validation size too large."

            _tensor_data = torch.utils.data.TensorDataset(preds[:validation_size],
                                                          labels[:validation_size])
            _models_preds_pool[x] = _tensor_data
        models_preds_pool = _models_preds_pool

    if diversity_strength is not None:
        esa_out = esa(models_preds_pool, 'loss', M, div_strength=diversity_strength)
    else:
        esa_out = esa(models_preds_pool, 'loss', M)
    return esa_out
    # if 'weights' in esa_out.keys():
    #     model_ids_to_ensemble = esa_out['models_chosen']
    #     weights = esa_out['weights']
    #     return {'models_chosen':}
    # else:
    #     model_ids_to_ensemble = esa_out['models_chosen']
    #     return model_ids_to_ensemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--esa",
        type=str,
        default="beam_search",
        help="Ensemble selection algorithm. See nes/ensemble_selection/esas.py. Default: beam_search.",
    )
    parser.add_argument("--M", type=int, default=5,
                        help="Ensemble size. Default: 5.")
    parser.add_argument(
        "--validation_size",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--diversity_strength",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max_seed",
        type=int,
        required=True
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == "cuda":
        torch.cuda.set_device(device)

    ENSEMBLE_SAVE_DIR = "saved_ensembles"
    Path(ENSEMBLE_SAVE_DIR).mkdir(exist_ok=True)

    # ===============================
    pool_name = "own_rs"
    SCHEMES = [pool_name]

    POOLS = {
        scheme: [model_seeds(arch=seed, init=seed, scheme=scheme) for seed in range(args.max_seed)]
        for scheme in SCHEMES
    }

    # ===============================

    BASELEARNER_DIR = "saved_model"

    pool_keys = POOLS[pool_name]

    pool = {
        k: load_baselearner(
            model_id=k,
            load_nn_module=False,
            baselearner_dir=BASELEARNER_DIR
        )
        for k in pool_keys
    }
    logging.info("Loaded baselearners")

    for baselearner in pool.values():  # move everything to right device
        baselearner.partially_to_device(device=args_to_device(device if str(device) != "cpu" else -1))
        # model.to_device(args_to_device(args.device))

    pools, num_arch_samples = get_pools_and_num_arch_samples(POOLS, pool_name, args.max_seed).values()

    esa = esas_registry[args.esa]
    severities = range(1)  # In case of our dataset

    result = defaultdict(list)
    result_weights = defaultdict(list)

    id_set = set()

    for i, pool_ids in enumerate(pools):
        for severity in severities:
            logging.info("Severity: {}".format(severity))
            population = {k: pool[k] for k in pool_ids}

            ens_chosen = run_esa(
                M=args.M, population=population, esa=esa, val_severity=severity,
                validation_size=args.validation_size,
                diversity_strength=None if args.esa != "beam_search_with_div" else args.diversity_strength
            )
            # print(ens_chosen)
            if (severity == 0) and (i == len(pools) - 1):
                id_set.update(
                    set([x.arch for x in ens_chosen['models_chosen']]))

            result[str(severity)].append(ens_chosen['models_chosen'])
            if "weights" in ens_chosen.keys():
                result_weights[str(severity)].append(ens_chosen['weights'])

        logging.info(f"Done {i + 1}/{len(pools)} for '{pool_name}', M={args.M}, esa={args.esa}, device={device}.")
        logging.info(f"Selected model IDs for ensemble: {id_set}")

    torch.save(id_set, os.path.join(ENSEMBLE_SAVE_DIR, f'ensemble_{args.M}_baselearners.pt'))

    if args.esa == "beam_search_with_div":
        args.esa = args.esa + f"_{args.diversity_strength}"

    if len(result_weights) > 0:
        to_dump = {"ensembles_chosen": result,
                   "num_arch_samples": num_arch_samples, "ensemble_weights": result_weights}
    else:
        to_dump = {"ensembles_chosen": result,
                   "num_arch_samples": num_arch_samples}

    if args.validation_size > -1:
        save_name = f"ensembles_chosen__esa_{args.esa}_M_{args.M}_pool_{pool_name}_valsize_{args.validation_size}.pickle"
    else:
        save_name = f"ensembles_chosen__esa_{args.esa}_M_{args.M}_pool_{pool_name}.pickle"

    with open(
            os.path.join(
                ENSEMBLE_SAVE_DIR,
                save_name
            ),
            "wb",
    ) as f:
        pickle.dump(to_dump, f)

    logging.info("Ensemble selection completed.")


def get_pools_and_num_arch_samples(POOLS, pool_name, max_seed):
    pool_keys = POOLS[pool_name]

    num_arch_samples = range(1)
    pool_at_samples = [pool_keys]

    return {"pools_at_samples": pool_at_samples, "num_arch_samples": num_arch_samples}


if __name__ == '__main__':
    enable_logging()
    main()
