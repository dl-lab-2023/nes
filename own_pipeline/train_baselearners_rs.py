import argparse
import logging
import os
from pathlib import Path
import random
import time
import json
from typing import Tuple
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
import openml
import torch
import torch.nn as nn
from ConfigSpace import ConfigurationSpace, Float, Configuration, Integer
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes
from own_pipeline.float_tabular_dataset import FloatTabularDataset
from openml import OpenMLTask
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from nes.ensemble_selection.containers import METRICS
from nes.ensemble_selection.utils import make_predictions, evaluate_predictions
from own_pipeline.containers.baselearner import model_seeds
from own_pipeline.lookahead import Lookahead
from own_pipeline.util import enable_logging


def set_seed_for_random_engines(seed: int, device):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    if device != "cpu":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        # see docs, set false to disable NON-deterministic algorithms
        torch.backends.cudnn.benchmark = False


def sample_random_hp_configuration(seed: int) -> Configuration:
    cs = ConfigurationSpace({
        "batch_normalization": [True, False],
        "stochastic_weight_avg": [True, False],
        "look_ahead_optimizer": [True, False],
        "LA_step_size": Float("LA_step_size", bounds=(0.5, 0.8)),
        "LA_num_steps": Integer("LA_num_steps", bounds=(5, 10)),
        "weight_decay": Float("weight_decay", bounds=(0.00001, 0.1), log=True),

        "learning_rate": Float("learning_rate", bounds=(0.0001, 0.1), log=True),
        "optimizer": ["SGD", "Adam", "AdamW"],
        "num_epochs": [100]
    })
    cs.seed(seed)
    return cs.sample_configuration(None)


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_norm: bool):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = self.get_layer_by_batchnorm(input_size, hidden_size, batch_norm)
        self.fc2 = self.get_layer_by_batchnorm(hidden_size, int(hidden_size // 2), batch_norm)
        self.fc3 = self.get_layer_by_batchnorm(int(hidden_size // 2), output_size, batch_norm)

    @staticmethod
    def get_layer_by_batchnorm(input_size: int, hidden_size: int, batch_norm: bool):
        if not batch_norm:
            return nn.Linear(input_size, hidden_size, dtype=torch.float64)
        return nn.Sequential(
            nn.BatchNorm1d(input_size, dtype=torch.float64),
            nn.Linear(input_size, hidden_size, dtype=torch.float64)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Tabulartrain(nn.Module):
    def __init__(self, input_size: int, output_size: int, batch_norm: bool) -> None:
        super(Tabulartrain, self).__init__()
        hidden_size = self._get_hidden_size(input_size)
        self.model = MLP(input_size, hidden_size, output_size, batch_norm)

    def _get_hidden_size(self, input_size):
        # Adjust the factor based on your preference
        hidden_size = int(input_size * 0.5)
        return hidden_size

    def forward(self, x):
        self.model.forward(x)

    def base_learner_train_save(self, seed: int, config: Configuration, train_loader: DataLoader,
                                test_loader: DataLoader, save_path: str, device: torch.device, num_classes: int):
        learning_rate = config["learning_rate"]
        optim = config["optimizer"]
        wd = config["weight_decay"]

        num_epochs = config["num_epochs"]

        criterion = nn.BCEWithLogitsLoss()

        if optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=learning_rate, weight_decay=wd)
        elif optim == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=wd)
        elif optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=wd)
        else:
            raise NotImplementedError()

        if config["look_ahead_optimizer"]:
            optimizer = Lookahead(optimizer, la_steps=config["LA_num_steps"], la_alpha=config["LA_step_size"])

        if config["stochastic_weight_avg"]:
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(optimizer, swa_lr=0.05)

            # Typical value, see https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
            swa_start = 0.75 * num_epochs

        logging.info(f"Starting training with {config=}, {seed=}")

        start_time = time.time()
        self.model.train()
        # Train the model
        for epoch in tqdm(range(num_epochs)):
            for i, (data, labels) in enumerate(train_loader):
                data = data.to(device)
                labels = labels.to(device)
                labels = torch.unsqueeze(labels, 1)

                outputs = self.model(data)
                loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    x = torch.isnan(data).any()
                    y = torch.isnan(labels).any()
                    raise ValueError(f"Training failed. Loss is NaN. - data contains NaN: {x or y}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if config["stochastic_weight_avg"] and epoch > swa_start:
                    swa_model.update_parameters(self.model)
                    swa_scheduler.step()

        if config["stochastic_weight_avg"]:
            torch.optim.swa_utils.update_bn(train_loader, swa_model)

        logging.info(
            f'Training finished in {round(time.time() - start_time, 2)} sec')

        # save model
        model_save_dir = os.path.join(save_path, f"seed_{seed}")
        Path(model_save_dir).mkdir(exist_ok=True, parents=True)

        model_id = model_seeds(arch=seed, init=seed, scheme="own_rs")._asdict()
        torch.save(model_id, os.path.join(model_save_dir, "model_id.pt"))
        torch.save(self.model, os.path.join(model_save_dir, "nn_module.pt"))

        prediction_model = self.model
        if config["stochastic_weight_avg"]:
            prediction_model = swa_model

        preds = make_predictions(prediction_model, test_loader, device, num_classes)
        evaluation = evaluate_predictions(preds)
        evaluation = {
            METRICS.loss: evaluation["loss"],
            METRICS.accuracy: evaluation["acc"],
            METRICS.error: 1 - evaluation["acc"],
            METRICS.ece: evaluation["ece"],
        }
        torch.save(
            {"preds": preds, "evals": evaluation},
            os.path.join(model_save_dir, "preds_evals.pt"),
        )

        hps = {}
        for key, value in config.items():
            hps[key] = value

        with open(os.path.join(model_save_dir, f"train_performance.json"), 'w') as f:
            json.dump({
                "seed": seed,
                "total_duration": time.time() - start_time,
                "evaluation": evaluation,
                "hyperparams": hps
            }, f, sort_keys=True, indent=4)

        logging.info("Saved baselearner. Done.")

        return self.model


# Defining the dataset

def dataloader(seed, batch_size, openml_task_id, test_size: float = 0.2):
    task: OpenMLTask = openml.tasks.get_task(task_id=openml_task_id,
                                             download_data=False,
                                             download_qualities=False,
                                             download_features_meta_data=False)

    num_classes = len(getattr(task, "class_labels"))
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute,
    )

    # if isinstance(y[1], bool):
    #     y = y.astype('bool')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )

    # Create a validator object to make sure that the data provided by
    # the user matches the autopytorch requirements
    input_validator = TabularInputValidator(
        is_classification=True,
        # logger_port=self._logger_port,
    )

    # Fit input validator to check the provided data
    # Also, an encoder is fit to both train and test data,
    # to prevent unseen categories during inference
    input_validator.fit(X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)

    dataset = FloatTabularDataset(
        X=X_train,
        Y=y_train,
        X_test=X_test,
        Y_test=y_test,
        validator=input_validator,
        resampling_strategy=HoldoutValTypes.holdout_validation,
        resampling_strategy_args=None,
        dataset_name=f"openml-{openml_task_id}",
        seed=seed
    )

    train_loader = DataLoader(dataset.get_dataset(
        split_id=0, train=True), batch_size=batch_size)
    test_loader = DataLoader(dataset.get_dataset(
        split_id=0, train=False), batch_size=batch_size)

    return train_loader, test_loader, X_train.shape, y_train.shape, num_classes


def get_layer_shape(shape: Tuple):
    if len(shape) == 1:
        return 1
    if len(shape) == 2:
        return shape[1]
    raise NotImplementedError()


# Define the train function to train
def run_train(seed: int, save_path: str, openml_task_id: int, only_download_dataset: bool):
    """
    Function that trains a given architecture and random hyperparameters.

    :param seed: (int) seed number used to seed all randomness
    :returns: None
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed_for_random_engines(seed, device)

    config = sample_random_hp_configuration(seed)

    train_loader, test_loader, X_train_shape, y_train_shape, num_classes = dataloader(
        seed, batch_size=16, openml_task_id=openml_task_id)

    if only_download_dataset:
        return

    input_size = get_layer_shape(X_train_shape)
    output_size = get_layer_shape(y_train_shape)

    model = Tabulartrain(input_size, output_size, config["batch_normalization"])
    model.to(device)

    model.base_learner_train_save(
        seed=seed,
        config=config,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        save_path=save_path,
        num_classes=num_classes
    )


if __name__ == '__main__':
    enable_logging()
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "--seed", type=int, default=1, help="Random generator seed")
    argParser.add_argument(
        "--openml_task_id", type=int, default=233088, help="OpenML task id")
    argParser.add_argument(
        "--only_download_dataset", type=bool, default=False, help="Only download the dataset and exit, do not train a base learner")
    args = argParser.parse_args()

    logging.info(f"Starting with args: {args}")

    save_path = f"./saved_model/task_{args.openml_task_id}"
    Path(save_path).mkdir(exist_ok=True, parents=True)

    run_train(args.seed, save_path=save_path, openml_task_id=args.openml_task_id, only_download_dataset=args.only_download_dataset)
