from autoPyTorch.datasets.tabular_dataset import TabularDataset
from typing import Tuple
import numpy as np


class FloatTabularDataset(TabularDataset):
    def __getitem__(self, index: int, train: bool = True) -> Tuple[np.ndarray, ...]:
        X, Y = super().__getitem__(index, train)
        X = X.astype(float)
        return X, Y
