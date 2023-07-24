from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.datasets.resampling_strategy import CrossValTypes, HoldoutValTypes, NoResamplingStrategyTypes
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms


class FloatTabularDataset(TabularDataset):
    def __init__(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 Y: Union[np.ndarray, pd.Series],
                 num_classes: int,
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 resampling_strategy: Union[CrossValTypes,
                                            HoldoutValTypes,
                                            NoResamplingStrategyTypes] = HoldoutValTypes.holdout_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 shuffle: Optional[bool] = True,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 dataset_name: Optional[str] = None,
                 validator: Optional[BaseInputValidator] = None,
                 ):
        super().__init__(X, Y, X_test, Y_test, resampling_strategy, resampling_strategy_args, shuffle, seed, train_transforms, val_transforms, dataset_name, validator)
        self.num_classes = num_classes

    def __getitem__(self, index: int, train: bool = True) -> Tuple[np.ndarray, ...]:
        X, Y = super().__getitem__(index, train)

        X = X.astype('float64')

        Y = Y.astype('int')
        Y = np.eye(self.num_classes)[Y]  # Onehot encode
        Y = Y.astype('float64')

        return X, Y
