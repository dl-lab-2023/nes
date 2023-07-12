from typing import List

import torch
import numpy as np

from torch.utils.data import TensorDataset
from collections import defaultdict
from types import SimpleNamespace
from itertools import combinations

from nes.ensemble_selection.containers import check_to_avoid_overwrite
from nes.ensemble_selection.utils import (
    evaluate_predictions,
    form_ensemble_pred,
    form_ensemble_pred_v_2,
)
from own_pipeline.baselearner import Baselearner

METRICS = SimpleNamespace(loss="loss", accuracy="acc", error="error", ece="ece")


class Ensemble:
    """
    A container class for ensembles which holds Baselearner objects. It can hold
    and compute the ensemble's predictions, evaluations.
    """

    def __init__(self, baselearners: List[Baselearner], bsl_weights=None):
        if len(set(b.device for b in baselearners)) != 1:
            raise ValueError("All baselearners should be on the same device.")

        if len(set(b.severities for b in baselearners)) != 1:
            raise ValueError(
                "All baselearners should be evaluated on the same number of severities."
            )

        self.baselearners = baselearners
        self.ensemble_size = len(self.baselearners)
        self.lsm_applied = True

        self.preds = None
        self.evals = None

        self.avg_baselearner_evals = None

        self.oracle_preds = None
        self.oracle_evals = None

        self.disagreement = None

        self.device = self.baselearners[0].device
        self._cpu_device = torch.device("cpu")
        self.severities = self.baselearners[0].severities

        self.repeating_bsls = len(set(b.model_id for b in self.baselearners)) != len(
            self.baselearners
        )

        if bsl_weights is not None:
            # bsl_weights list ordering should be consistent with the baselearners list ordering.
            assert not self.repeating_bsls, "Expected unique base learners if given weights."
            self.bsl_weights = torch.tensor(bsl_weights).to(self.device)
            assert bsl_weights.shape == (self.ensemble_size,), "Number of weights doesn't match number of base learners"
        else:
            self.bsl_weights = None

    def to_device(self, device=None):
        if device is None:
            device = self._cpu_device

        for b in self.baselearners:
            b.to_device(device)

        if self.preds is not None:
            for key, dct in self.preds.items():
                for k, tsr_dst in dct.items():
                    dct[k] = TensorDataset(
                        tsr_dst.tensors[0].to(device), tsr_dst.tensors[1].to(device)
                    )

        self.device = device

    @check_to_avoid_overwrite("preds")
    def compute_preds(self, combine_post_softmax=True):
        if not self.repeating_bsls:
            preds_dict = {
                b.model_id: b.preds.tensors[0]
                for b in self.baselearners
            }
            labels = (
                self.baselearners[0].preds.tensors[1]
            )

            preds = form_ensemble_pred(
                preds_dict,
                lsm_applied=False,
                combine_post_softmax=combine_post_softmax,
                bsl_weights=self.bsl_weights,
            )
            preds = TensorDataset(preds, labels)

            ens_preds = preds
        else:
            print("There are repeating base learners.")
            preds_list = [
                b.preds.tensors[0]
                for b in self.baselearners
            ]
            labels = (
                self.baselearners[0].preds.tensors[1]
            )

            preds = form_ensemble_pred_v_2(
                preds_list,
                lsm_applied=False,
                combine_post_softmax=combine_post_softmax,
            )
            preds = TensorDataset(preds, labels)

            ens_preds = preds

        self.preds = ens_preds

    @check_to_avoid_overwrite("evals")
    def compute_evals(self):
        if self.preds is None:
            raise ValueError(
                "Baselearners' predictions not available. Run .compute_preds(...) first."
            )

        evaluation = evaluate_predictions(self.preds, self.lsm_applied)
        evaluation = {
            METRICS.loss: evaluation["loss"],
            METRICS.accuracy: evaluation["acc"],
            METRICS.error: 1 - evaluation["acc"],
            METRICS.ece: evaluation["ece"],
        }

        self.evals = evaluation

    @check_to_avoid_overwrite("avg_baselearner_evals")
    def compute_avg_baselearner_evals(self, severities=None):
        evals = defaultdict(lambda: defaultdict(dict))

        for metric in METRICS.__dict__.values():
            avg = np.mean(
                [
                    b.evals[metric]
                    for b in self.baselearners
                ]
            )
            evals[metric] = avg

        self.avg_baselearner_evals = evals

    @check_to_avoid_overwrite("oracle_preds")
    def compute_oracle_preds(self):
        labels = (
            self.baselearners[0].preds.tensors[1]
        )  # labels of any baselearner are fine

        all_preds_list = [
            b.preds.tensors[0].softmax(1)
            for b in self.baselearners
        ]
        all_preds = torch.stack(all_preds_list, dim=1)

        _, oracle_selection = torch.max(
            all_preds[range(len(all_preds)), :, labels], 1
        )
        _oracle_preds = all_preds[
                        range(len(all_preds)), oracle_selection, :
                        ].log()

        oracle_preds = TensorDataset(
            _oracle_preds, labels
        )

        self.oracle_preds = oracle_preds

    @check_to_avoid_overwrite("oracle_evals")
    def compute_oracle_evals(self):
        if self.oracle_preds is None:
            raise ValueError(
                "Oracle's predictions not available. Run .compute_oracle_preds(...) first."
            )

        preds = self.oracle_preds
        evaluation = evaluate_predictions(preds, self.lsm_applied)
        evaluation = {
            METRICS.loss: evaluation["loss"],
            METRICS.accuracy: evaluation["acc"],
            METRICS.error: 1 - evaluation["acc"],
            METRICS.ece: evaluation["ece"],
        }

        self.oracle_evals = evaluation

    @check_to_avoid_overwrite("disagreement")
    def compute_disagreement(self):
        disagreement = defaultdict(dict)
        # pairs_of_baselearners = combinations(self.baselearners, 2)
        num_pairs = len(list(combinations(self.baselearners, 2)))

        for severity in self.severities:
            for data_type in ["val", "test"]:

                avg_disagreement = 0
                avg_norm_disagreement = 0
                for bsl1, bsl2 in combinations(self.baselearners, 2):
                    bsl1_class_preds = (
                        bsl1.preds[data_type][str(severity)].tensors[0].max(dim=1)[1]
                    )
                    bsl2_class_preds = (
                        bsl2.preds[data_type][str(severity)].tensors[0].max(dim=1)[1]
                    )

                    num_disagreements = (
                        (bsl1_class_preds != bsl2_class_preds).sum().item()
                    )
                    total = bsl1_class_preds.shape[0]

                    avg_acc = (
                                      bsl1.evals[data_type][str(severity)]["acc"]
                                      + bsl2.evals[data_type][str(severity)]["acc"]
                              ) / 2

                    avg_disagreement += num_disagreements / total
                    avg_norm_disagreement += num_disagreements / total / (1 - avg_acc)

                avg_disagreement /= num_pairs
                avg_norm_disagreement /= num_pairs

                disagreement[data_type][str(severity)] = {
                    "disagreement": avg_disagreement,
                    "normalized_disagreement": avg_norm_disagreement,
                }

        self.disagreement = disagreement
