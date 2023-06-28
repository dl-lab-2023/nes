import logging
import os
from pathlib import Path
import time

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from nes.tab_experiment.baselearner_train.re_sampler import sample_random_genotype_hp
from nes.tab_experiment.baselearner_train.utilities import get_data
from ConfigSpace.configuration_space import Configuration

import torch


def train_using_auto_pytorch(tab_task_id, seed, configuration, num_epochs, bslrn_batch_size,
                             logger, out_path='output', mode='train', debug=False,
                             n_workers=4, **kwargs):
    """Function that trains a given architecture.

    Args:
        TODO

    Returns:
        None
    """

    ############################################################################
    # Data Loading
    # ============
    start_time = time.time()

    X_train, X_test, y_train, y_test, resampling_strategy_args, categorical_indicator = get_data(
        task_id=tab_task_id,
        seed=seed,
    )

    print(f"Loaded dataset, X_train shape: {X_train.shape}")

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

    ############################################################################
    # Build and fit a classifier
    # ==========================
    nr_workers = 1

    api = TabularClassificationTask(
        temporary_directory=temp_dir,
        output_directory=output_dir,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        resampling_strategy=HoldoutValTypes.stratified_holdout_validation,
        resampling_strategy_args=resampling_strategy_args,
        ensemble_size=1,
        ensemble_nbest=1,
        max_models_on_disc=10,
        include_components=include_updates,
        search_space_updates=search_space_updates,
        seed=args.seed,
        n_jobs=nr_workers,
        n_threads=args.nr_threads,
    )

    ############################################################################
    # Refit on the sampled hp configuration
    # ==================================
    input_validator = TabularInputValidator(
        is_classification=True,
    )
    input_validator.fit(
        X_train=X_train.copy(),
        y_train=y_train.copy(),
        X_test=X_test.copy(),
        y_test=y_test.copy(),
    )

    dataset = TabularDataset(
        X=X_train,
        Y=y_train,
        X_test=X_test,
        Y_test=y_test,
        seed=args.seed,
        validator=input_validator,
        resampling_strategy=NoResamplingStrategyTypes.no_resampling,
    )
    dataset.is_small_preprocess = False
    print(f"Fitting pipeline with {args.epochs} epochs")

    # Train the model
    fitted_pipeline, run_info, run_value, dataset = api.fit_pipeline(
        configuration=configuration,
        budget_type='epochs',
        budget=args.epochs,
        dataset=dataset,
        run_time_limit_secs=args.func_eval_time,
        eval_metric='balanced_accuracy',
        memory_limit=12000,
    )

    # Validation
    X_train = dataset.train_tensors[0]  # TODO why only for the first item?
    y_train = dataset.train_tensors[1]
    X_test = dataset.test_tensors[0]
    y_test = dataset.test_tensors[1]

    train_predictions = fitted_pipeline.predict(X_train)
    test_predictions = fitted_pipeline.predict(X_test)

    train_balanced_accuracy = metrics.balanced_accuracy(
        y_train, train_predictions.squeeze())
    test_balanced_accuracy = metrics.balanced_accuracy(
        y_test, test_predictions.squeeze())
    duration = time.time() - start_time

    print(f'Final Train Balanced accuracy: {train_balanced_accuracy}')
    print(f'Final Test Balanced accuracy: {test_balanced_accuracy}')
    print(f'Time taken: {duration}')

    # TODO store metrics


def get_auto_pytorch_configuration(hp_config):
    # We're following https://automl.github.io/Auto-PyTorch/master/examples/40_advanced/example_single_configuration.html#sphx-glr-examples-40-advanced-example-single-configuration-py

    # Use configspaces lib
    # TODO fix the import of "Configuration"!
    # This is the library code used by the requirements.txt, with the used version from 2021:
    # https://github.com/automl/ConfigSpace/tree/f3b5ece99ff132c56c8d741a933a82d3bdc3f635
    configuration = Configuration()

    return configuration


# TODO replace this function by a cluster script
if __name__ == "__main__":
    hp_id, hp_config = sample_random_genotype_hp()

    auto_pytorch_configuration = get_auto_pytorch_configuration(hp_config)

    train_using_auto_pytorch(1, 1, auto_pytorch_configuration, 1, 1, None)


# TODO is there a possibility to let auto pytorch do the sampling? see
# https://automl.github.io/Auto-PyTorch/master/examples/40_advanced/example_custom_configuration_space.html#sphx-glr-examples-40-advanced-example-custom-configuration-space-py
# this looks good, but we need to check whether it's possible to parallelize this with SLURM
