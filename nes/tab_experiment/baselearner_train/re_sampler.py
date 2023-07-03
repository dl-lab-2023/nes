from typing import Any, Dict, Tuple
import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration


def sample_random_genotype_hp() -> Tuple[str, Dict[str, Any]]:
    config_space = ConfigurationSpace(
        name="experiment-configspace",
        space={
            "AdamWOptimizer:lr": (0.001, 0.1),  # UniformFloat
            "CosineAnnealingWarmRestarts:n_restarts": (1, 10),  # UniformInt
        }
    )
    return config_space.sample_configuration(size=1)
