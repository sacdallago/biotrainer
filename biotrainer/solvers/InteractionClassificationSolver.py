from typing import Dict, Union

import torch

from .Solver import Solver
from .SequenceClassificationSolver import SequenceClassificationSolver


class InteractionClassificationSolver(SequenceClassificationSolver, Solver):

    def _logits_to_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        prediction_probabilities = torch.sigmoid(logits)
        predicted_classes = torch.round(prediction_probabilities)

        return predicted_classes.int()

    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:
        return network_output.squeeze()

    def _compute_metrics(
            self, predicted: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Union[int, float]]:
        return super()._compute_metrics(predicted, labels.int())
