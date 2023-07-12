from nes.ensemble_selection.utils import make_predictions, evaluate_predictions
from collections import defaultdict, namedtuple
from pathlib import Path
from torch.utils.data import TensorDataset
import os
import torch

from nes.ensemble_selection.containers import METRICS, check_to_avoid_overwrite

model_seeds = namedtuple(typename="model_seeds", field_names=["arch", "init", "scheme"])


class Baselearner:
    """
    A container class for baselearner networks which can hold the nn.Module,
    predictions (as tensors) and evaluations. It has methods for computing the predictions
    and evaluations on validation and test sets with shifts of varying
    severities.
    """

    _cpu_device = torch.device("cpu")

    def __init__(
            self, model_id, severities, device=None, nn_module=None, preds=None, evals=None,
    ):
        self.model_id = model_id
        self.device = device
        self.nn_module = nn_module
        self.preds = preds
        self.evals = evals
        self.lsm_applied = False
        self.severities = severities

    def to_device(self, device=None):
        if device is None:
            device = self._cpu_device

        if self.nn_module is not None:
            self.nn_module.to(device)

        if self.preds is not None:
            self.preds = TensorDataset(
                self.preds.tensors[0].to(device), self.preds.tensors[1].to(device)
            )

        self.device = device

    def partially_to_device(self, device=None):
        if device is None:
            device = self._cpu_device

        if self.nn_module is not None:
            self.nn_module.to(device)

        if self.preds is not None:
            self.preds = TensorDataset(
                self.preds.tensors[0].to(device), self.preds.tensors[1].to(device)
            )

        self.device = device

    @check_to_avoid_overwrite("preds")
    def compute_preds(self, dataloaders, severities=None, num_classes=10):
        """
        Computes and stores the predictions of the model as tensors.

        Args:
            dataloaders (dict): Contains dataloaders for datasets over which to make predictions. See e.g.
                `create_dataloader_dict_fmnist` in `nes/ensemble_selection/utils.py`.
            severities (list-like, optional): Severity levels (as ints) of data shift.
            num_classes (int, optional): Number of classes.
        """
        if severities is None:
            severities = self.severities

        if dataloaders["metadata"]["device"] != self.device:
            raise ValueError(
                f'Data is on {dataloaders["metadata"]["device"]}, but baselearner is on {self.device}'
            )

        preds = defaultdict(dict)

        for severity in severities:
            for data_type in ["val", "test"]:
                loader = dataloaders[data_type][str(severity)]
                preds[data_type][str(severity)] = make_predictions(
                    self.nn_module, loader, self.device, num_classes=num_classes
                )

        self.preds = preds

    @check_to_avoid_overwrite("evals")
    def compute_evals(self, severities=None):
        if severities is None:
            severities = self.severities

        if self.preds is None:
            raise ValueError(
                "Baselearner predictions not available. Run .compute_preds(...) first."
            )

        evals = defaultdict(dict)

        for severity in severities:
            for data_type in ["val", "test"]:
                preds = self.preds[data_type][str(severity)]
                evaluation = evaluate_predictions(preds, self.lsm_applied)
                evaluation = {
                    METRICS.loss: evaluation["loss"],
                    METRICS.accuracy: evaluation["acc"],
                    METRICS.error: 1 - evaluation["acc"],
                    METRICS.ece: evaluation["ece"],
                }

                evals[data_type][str(severity)] = evaluation

        self.evals = evals

    def save(self, directory, force_overwrite=False):
        self.to_device(self._cpu_device)

        dir = os.path.join(
            directory,
            f"arch_{self.model_id.arch}_init_{self.model_id.init}_scheme_{self.model_id.scheme}",
        )

        if force_overwrite:
            Path(dir).mkdir(parents=True, exist_ok=True)
            print(f"Forcefully overwriting {dir}")
        else:
            Path(dir).mkdir(parents=True, exist_ok=False)

        torch.save(self.model_id._asdict(), os.path.join(dir, "model_id.pt"))

        if self.nn_module is not None:
            torch.save(self.nn_module, os.path.join(dir, "nn_module.pt"))

        torch.save(
            {"preds": self.preds, "evals": self.evals},
            os.path.join(dir, "preds_evals.pt"),
        )

    @classmethod
    def load(cls, dir, load_nn_module=False):
        loaded = torch.load(os.path.join(dir, "model_id.pt"))
        model_id = model_seeds(**loaded)

        preds, evals = torch.load(os.path.join(dir, "preds_evals.pt")).values()

        if load_nn_module:
            nn_module = torch.load(os.path.join(dir, "nn_module.pt"))
        else:
            nn_module = None

        # find number of severities
        num_sevs = len(preds)
        severities = range(num_sevs)

        device = cls._cpu_device
        obj = cls(
            model_id=model_id,
            severities=severities,
            device=device,
            nn_module=nn_module,
            preds=preds,
            evals=evals,
        )

        return obj


def load_baselearner(model_id, load_nn_module, baselearner_dir) -> Baselearner:
    dir = os.path.join(
        baselearner_dir,
        f"seed_{model_id[0]}",
    )
    to_return = Baselearner.load(dir, load_nn_module)
    # assert to_return.model_id == model_id
    return to_return
