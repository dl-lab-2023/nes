import argparse
import logging
import os
import random
import time
from typing import Tuple

import numpy as np
import openml
import torch
import torch.nn as nn
from ConfigSpace import ConfigurationSpace, Float, Configuration
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        "learning_rate": Float("learning_rate", bounds=(0.0001, 0.1), log=True),
        "weight_decay": Float("weight_decay", bounds=(0.001, 0.1), log=True),
        "optimizer": ["SGD", "Adam"],
        "num_epochs": [100]
    })
    cs.seed(seed)
    return cs.sample_configuration(None)


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size // 2), dtype=torch.float64)
        self.fc3 = nn.Linear(int(hidden_size // 2), output_size, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Tabulartrain(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(Tabulartrain, self).__init__()
        hidden_size = self._get_hidden_size(input_size)
        self.model = MLP(input_size, hidden_size, output_size)

    def _get_hidden_size(self, input_size):
        # Adjust the factor based on your preference
        hidden_size = int(input_size * 0.5)
        return hidden_size

    def forward(self, x):
        self.model.forward(x)

    def base_learner_train_save(self, seed: int, config: Configuration, train_loader: DataLoader,
                                test_loader: DataLoader, save_path: str, device: torch.device):
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
        else:
            raise NotImplementedError()

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
                    raise ValueError("Training failed. Loss is NaN.")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        logging.info(
            f'Training completed for model (config space {config}, seed {seed}) ' +
            f'in {round(time.time() - start_time, 2)}')

        self.model.eval()
        total_loss = 0.0
        for epoch in tqdm(range(num_epochs)):
            for i, (data, labels) in enumerate(test_loader):
                data = data.to(device)
                labels = labels.to(device)
                labels = torch.unsqueeze(labels, 1)

                outputs = self.model(data)
                loss = criterion(outputs, labels)
                total_loss += loss

        logging.info(
            f'Evaluation completed for model (config space {config}, seed {seed}) ' +
            f'in {round(time.time() - start_time, 2)}')

        model_save_path = os.path.join(
            save_path, f"seed_{seed}_loss_{total_loss}_config_{config}.pt"
        )
        torch.save(self.model.state_dict(), model_save_path)
        logging.info(
            f'Saved model (arch {config}, seed {seed}) ' +
            f'after epoch {num_epochs} in {round(time.time() - start_time, 2)} secs.')

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

    # Fit input validator to check the provided data
    # Also, an encoder is fit to both train and test data,
    # to prevent unseen categories during inference
    input_validator.fit(X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)

    dataset = TabularDataset(
        X=X_train,
        Y=y_train,
        X_test=X_test,
        Y_test=y_test,
        validator=input_validator,
        resampling_strategy=HoldoutValTypes.holdout_validation,
        resampling_strategy_args=None,
        dataset_name=f"openml-{task_id}",
        seed=seed
    )

    train_loader = DataLoader(dataset.get_dataset(split_id=0, train=True), batch_size=batch_size)
    test_loader = DataLoader(dataset.get_dataset(split_id=0, train=False), batch_size=batch_size)

    return train_loader, test_loader, X_train.shape, y_train.shape


def get_layer_shape(shape: Tuple):
    if len(shape) == 1:
        return 1
    if len(shape) == 2:
        return shape[1]
    raise NotImplementedError()


# Define the train function to train
def run_train(seed: int, save_path: str):
    """
    Function that trains a given architecture and random hyperparameters.

    :param seed: (int) seed number used to seed all randomness
    :returns: None
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed_for_random_engines(seed, device)

    config = sample_random_hp_configuration(seed)

    train_loader, test_loader, X_train_shape, y_train_shape = dataloader(
        seed, batch_size=16)

    input_size = get_layer_shape(X_train_shape)
    output_size = get_layer_shape(y_train_shape)

    model = Tabulartrain(input_size, output_size)
    model.to(device)

    logging.info(f" (configurations {config}, seed: {seed})...")

    model.base_learner_train_save(
        seed=seed,
        config=config,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        save_path=save_path,
    )


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--seed", type=int, required=True, help="Random generator seed")
    args = argParser.parse_args()

    save_path = "./saved_model"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    run_train(args.seed, save_path=save_path)

    print("Done")
