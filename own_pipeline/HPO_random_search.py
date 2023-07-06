import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import os
from pathlib import Path
import sys
import logging
import pandas
import random

from sklearn.model_selection import train_test_split

import openml
from torch.utils.data import DataLoader, TensorDataset

from ConfigSpace import ConfigurationSpace, Float, Configuration

from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes
from autoPyTorch.datasets.tabular_dataset import TabularDataset


def seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # see docs, set to false to disable NON-deterministic algorithms
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def configurations(seed: int) -> Configuration:

    cs = ConfigurationSpace({
        "learning_rate": Float("learning_rate", bounds=(0.0001, 0.1), log=True),
        "weight_decay": Float("weight_decay", bounds=(0.001, 0.1), log=True),
        "optimizer": ["SGD", "Adam"],
        "num_epochs": [100]
    })
    cs.seed(seed)
    return cs.sample_configuration(None)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size//2))
        self.fc3 = nn.Linear(int(hidden_size//2), 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Tabulartrain(nn.Module):
    def __init__(self, input_size) -> None:
        super(Tabulartrain, self).__init__()
        hidden_size = self._get_hidden_size(input_size)
        self.model = MLP(input_size, hidden_size)

    def _get_hidden_size(self, input_size):
        # Adjust the factor based on your preference
        hidden_size = int(input_size * 0.5)
        return hidden_size

    def forward(self, x):
        self.model.forward(x)

    def base_learner_train_save(self, seed, config_space, train_loader, test_loader,
                                save_path, device):

        learning_rate = config_space["learning_rate"]
        optim = config_space["optimizer"]
        wd = config_space["weight_decay"]

        num_epochs = config_space["num_epochs"]

        criterion = nn.NLLLoss()

        if optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=learning_rate, weight_decay=wd)
        elif optim == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=wd)

        start_time = time.time()
        torch.manual_seed(0)

        # Train the model
        for epoch in range(num_epochs):
            for i, (data, labels) in enumerate(train_loader):

                data = data.to(device)
                labels = labels.to(device)

                outputs = self.model(data)
                loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    raise ValueError("Training failed. Loss is NaN.")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        logging.info('Training completed for model (config space {}, init {}) in {} '
                     'secs.'.format(config_space, round(time.time() -
                                                        start_time, 2)))

        model_save_path = os.path.join(
            save_path, f"config_space_{config_space}_init_{seed}_epoch_{num_epochs}.pt"
        )
        torch.save(self.model.state_dict(), model_save_path)
        logging.info(
            'Saved model (arch {}, init {}) after epoch {} in {} '
            'secs.'.format(config_space, seed, num_epochs, round(time.time()
                                                                 - start_time, 2)))

        return self.model


# Defining the dataset

def dataloader(seed, batch_size, task_id=233088, test_size: float = 0.2):
    task = openml.tasks.get_task(task_id=task_id)
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

    # Fit a input validator to check the provided data
    # Also, an encoder is fit to both train and test data,
    # to prevent unseen categories during inference
    input_validator.fit(X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)

    dataset = TabularDataset(
        X=X_train, Y=y_train,
        X_test=X_test, Y_test=y_test,
        validator=input_validator,
        resampling_strategy=HoldoutValTypes.holdout_validation,
        resampling_strategy_args=None,
        dataset_name=f"openml-{task_id}",
        seed=seed
    )

    train_loader = DataLoader(dataset)
    test_loader = DataLoader(dataset.test_tensors)

    return train_loader, test_loader, X_train.shape, y_train.shape


# Define the train function to train
def run_train(seed):
    """Function that trains a given architecture.

    Args:
        seed                 (int): seed number
        arch_id              (int): architecture id
        arch                 (str): architecture genotype as string
        num_epochs           (int): number of epochs to train
        bslrn_batch_size     (int): mini-batch size
        exp_name             (str): directory where to save results
        logger    (logging.Logger): logger object
        data_path            (str): directory where the dataset is stored
        mode                 (str): train or validation
        debug               (bool): train for a single mini-batch only
        dataset              (str): dataset name
        global_seed          (int): global seed for optimizer runs

    Returns:
        None
    """

    seeds(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_space = configurations(seed)

    train_loader, test_loader, X_train_shape, y_train_shape = dataloader(
        seed, batch_size=16)

    if len(y_train_shape) == 1:
        model_out_shape = 1
    elif len(y_train_shape) == 2:
        model_out_shape = y_train_shape[1]
    else:
        raise NotImplementedError()

    model = Tabulartrain(model_out_shape)
    model.to(device)

    logging.info(f" (configurations {config_space}, init: {seed})...")

    model.base_learner_train_save(
        seed=seed,
        config_space=config_space,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        save_path='./saved_model',
    )

# Get the outputs and print them


if __name__ == '__main__':

    run_train(seed=1)