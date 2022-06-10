import torch

from typing import Optional, Dict, Union

from .Solver import Solver


class SequenceClassificationSolver(Solver):

    def _logits_to_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        prediction_probabilities = torch.softmax(logits, dim=1)
        _, predicted_classes = torch.max(prediction_probabilities, dim=1)

        return predicted_classes

    def _compute_metrics(
          self, predicted: torch.Tensor, labels: torch.Tensor, masks: Optional[torch.BoolTensor] = None
    ) -> Dict[str, Union[int, float]]:

        return {
            'accuracy': ((predicted == labels).float().sum() / len(labels)).item()
        }
