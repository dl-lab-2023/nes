import logging
import os
from pathlib import Path
from nes.tab_experiment.baselearner_train.utilities import get_data

import torch

from nes.darts.baselearner_train.model import model_tab as model_tab


def run_train(seed, hp_id, arch, num_epochs, bslrn_batch_size, exp_name,
              logger, data_path='data', out_path='output', mode='train', debug=False,
              global_seed=0, n_workers=4,
              hp_config={}, **kwargs):
    """Function that trains a given architecture.

    Args:
        seed                 (int): seed number
        hp_id                (int): hyperparameter combination id
        arch                 (str): architecture genotype as string
        num_epochs           (int): number of epochs to train
        bslrn_batch_size     (int): mini-batch size
        exp_name             (str): directory where to save results
        logger    (logging.Logger): logger object
        data_path            (str): directory where the dataset is stored
        mode                 (str): train or validation
        debug               (bool): train for a single mini-batch only
        global_seed          (int): global seed for optimizer runs
        n_workers            (int): number of workers for dataloaders

    Returns:
        None
    """
    device = torch.device(f'cuda:0')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    Path(exp_name).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(os.path.join(exp_name,
                                          f"hp{hp_id}.log"), mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    genotype = eval(arch)
    
    tab_task_id = 1 # TODO ######################################

    ############################################################################
    # Data Loading
    # ============
    X_train, X_test, y_train, y_test, resampling_strategy_args, categorical_indicator = get_data(
        task_id=tab_task_id,
        seed=seed,
    )

    # TODO read Tab paper: is this required for us?
    pipeline_update, search_space_updates, include_updates = get_updates_for_regularization_cocktails(
        categorical_indicator,
        args,
    )

    output_dir = os.path.expanduser(
        os.path.join(
            out_path,
            f'{seed}',
            f'{tab_task_id}',
            f'{tab_task_id}_out',
        )
    )
    temp_dir = os.path.expanduser(
        os.path.join(
            out_path,
            f'{seed}',
            f'{tab_task_id}',
            f'{tab_task_id}_tmp',
        )
    )

    logger.info(f"[{mode}] (hp {hp_id}: {genotype}, init: {seed})...")

    # TODO update parameters in function call
    model_tab.base_learner_train_save(seed_init=seed,
                                       hp_id=hp_id,
                                       genotype=genotype,
                                       X_train=X_train,
                                       X_test=X_test,
                                       y_train=y_train,
                                       y_test=y_test,
                                       num_epochs=num_epochs,
                                       save_path=exp_name,
                                       device=device,
                                       dataset=dataset,
                                       verbose=True,
                                       debug=debug,
                                       global_seed=global_seed,
                                       anchor=anchor,
                                       anch_coeff=anch_coeff,
                                       logger=logger,
                                       hp_config=hp_config,
                                       **kwargs)
