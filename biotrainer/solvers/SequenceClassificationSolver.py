import torch

from .Solver import Solver
from .ClassificationSolver import ClassificationSolver


class SequenceClassificationSolver(ClassificationSolver, Solver):

    def _logits_to_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        prediction_probabilities = torch.softmax(logits, dim=1)
        _, predicted_classes = torch.max(prediction_probabilities, dim=1)

        return predicted_classes
