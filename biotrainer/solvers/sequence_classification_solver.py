import torch

from .solver import Solver


class SequenceClassificationSolver(Solver):

    def _logits_to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=1)

    def _probabilities_to_predictions(self, probabilities: torch.Tensor) -> torch.Tensor:
        _, predicted_classes = torch.max(probabilities, dim=1)

        return predicted_classes
