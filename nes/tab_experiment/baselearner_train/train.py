import logging
import os
from pathlib import Path
from nes.tab_experiment.baselearner_train.re_sampler import sample_random_genotype_hp
from nes.tab_experiment.baselearner_train.utilities import get_data

import torch


def run_train(tab_task_id, seed, hp_id, hp_config, num_epochs, bslrn_batch_size,
              logger, out_path='output', mode='train', debug=False,
              n_workers=4, **kwargs):
    """Function that trains a given architecture.

    Args:
        TODO

    Returns:
        None
    """

    # Data loader
    X_train, X_test, y_train, y_test, resampling_strategy_args, categorical_indicator = get_data(
        task_id=tab_task_id,
        seed=seed,
    )

    print(X_train.shape)



hp_id, hp_config = sample_random_genotype_hp()

run_train(1, 1, 1, hp_config, 1, 1, None)
