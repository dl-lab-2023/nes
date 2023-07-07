import os
import dill as pickle
import torch
import time
import argparse
from collections import defaultdict
from pathlib import Path

from nes.ensemble_selection.utils import (
    model_seeds,
    args_to_device,
    POOLS,
    SCHEMES,
)
from nes.ensemble_selection.config import BUDGET, PLOT_EVERY
from nes.ensemble_selection.containers import Ensemble, Baselearner
from nes.ensemble_selection.esas import registry as esas_registry, run_esa


def load_baselearner(model_id, load_nn_module, baselearner_dir):
    dir = os.path.join(
        baselearner_dir,
        f"seed_{model_id}",
    )
    to_return = Baselearner.load(dir, load_nn_module)
    # assert to_return.model_id == model_id
    return to_return


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

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    ENSEMBLE_SAVE_DIR = "saved_ensembles"
    Path(ENSEMBLE_SAVE_DIR).mkdir(exist_ok=True)

    # ===============================

    if args.pool_name == "nes_rs_esa":
        POOLS.update(
            {
                scheme: [model_seeds(arch=args.arch_id, init=seed, scheme=scheme)
                         for seed in range(BUDGET)]
                for scheme in SCHEMES
                if scheme == "nes_rs_esa"
            }
        )

    # ===============================

    BASELEARNER_DIR = "saved_model"

    pool_keys = POOLS[args.pool_name]

    pool = {
        k: load_baselearner(
            model_id=k,
            load_nn_module=False,
            baselearner_dir=BASELEARNER_DIR
        )
        for k in pool_keys
    }
    print("Loaded baselearners")

    for model in pool.values():  # move everything to right device
        model.partially_to_device(
            data_type='val', device=args_to_device(args.device))
        # model.to_device(args_to_device(args.device))

    pools, num_arch_samples = get_pools_and_num_arch_samples(
        args.pool_name).values()

    esa = esas_registry[args.esa]
    severities = range(6) if (
        args.dataset in ["cifar10", "cifar100", "tiny"]) else range(1)

    result = defaultdict(list)
    result_weights = defaultdict(list)

    id_set = set()

    for i, pool_ids in enumerate(pools):
        for severity in severities:
            print("Severity: {}".format(severity))
            population = {k: pool[k] for k in pool_ids}

            ens_chosen = run_esa(
                M=args.M, population=population, esa=esa, val_severity=severity,
                validation_size=args.validation_size, diversity_strength=None if args.esa != "beam_search_with_div" else args.diversity_strength
            )
            # print(ens_chosen)
            if (severity == 0) and (i == len(pools) - 1):
                id_set.update(
                    set([x.arch for x in ens_chosen['models_chosen']]))

            result[str(severity)].append(ens_chosen['models_chosen'])
            if "weights" in ens_chosen.keys():
                result_weights[str(severity)].append(ens_chosen['weights'])

        print(
            "Done {}/{} for {}, M = {}, esa = {}, device = {}.".format(i+1, len(pools),
                                                                       args.pool_name, args.M,
                                                                       args.esa, args.device)
        )
        print(id_set)

    torch.save(id_set, os.path.join(
        ENSEMBLE_SAVE_DIR, 'ids_{}.pt'.format(args.M)))

    if args.esa == "beam_search_with_div":
        args.esa = args.esa + f"_{args.diversity_strength}"

    if len(result_weights) > 0:
        to_dump = {"ensembles_chosen": result,
                   "num_arch_samples": num_arch_samples, "ensemble_weights": result_weights}
    else:
        to_dump = {"ensembles_chosen": result,
                   "num_arch_samples": num_arch_samples}

    if args.validation_size > -1:
        save_name = f"ensembles_chosen__esa_{args.esa}_M_{args.M}_pool_{args.pool_name}_valsize_{args.validation_size}.pickle"
    else:
        save_name = f"ensembles_chosen__esa_{args.esa}_M_{args.M}_pool_{args.pool_name}.pickle"

    with open(
        os.path.join(
            ENSEMBLE_SAVE_DIR,
            save_name
        ),
        "wb",
    ) as f:
        pickle.dump(to_dump, f)

    print("Ensemble selection completed.")


def get_pools_and_num_arch_samples(pool):
    pool_keys = POOLS[pool]
    if args.pool_name not in ["deepens_darts", "deepens_amoeba"]:
        assert len(pool_keys) == BUDGET

        num_arch_samples = range(PLOT_EVERY, BUDGET + 1, PLOT_EVERY)
        pool_at_samples = [pool_keys[:num_samples]
                           for num_samples in num_arch_samples]
    else:
        num_arch_samples = range(1)
        pool_at_samples = [pool_keys]

    return {"pools_at_samples": pool_at_samples, "num_arch_samples": num_arch_samples}


if __name__ == '__main__':
    main()
