from typing import Any, Dict, Tuple
import numpy as np


def sample_random_genotype_hp() -> Tuple[str, Dict[str, Any]]:
    """
    returns random hps every time you call the function
    :returns: string-id, dictionary with hps
    """
    sampled_hps = {
        'lr': np.random.choice([0.1, 0.01, 0.001]),
        # https://pytorch.org/docs/stable/optim.html
        'optimizer': np.random.choice(['Adam', 'SGD', 'RMSprop']),
        'sgd-momentum': np.random.choice([0.9]),
        'sgd-weight-decay': np.random.choice([3e-5])
    }

    id = ""
    for hp_name, hp_value in sampled_hps.items():
        id = id + f"{hp_name}-{hp_value}|"

    id = id[:-1]  # Remove last "|" character

    return id, sampled_hps
