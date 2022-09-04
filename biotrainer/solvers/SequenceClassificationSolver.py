import torch

from typing import Dict, Union
from torchmetrics import Accuracy, Precision, Recall, F1Score

from .Solver import Solver


class SequenceClassificationSolver(Solver):

    def __init__(self, *args, **kwargs):
        super(SequenceClassificationSolver, self).__init__(*args, **kwargs)
        # Init metrics
        num_classes = kwargs['num_classes']
        self.acc = Accuracy(average="micro", num_classes=num_classes)
        self.precision = Precision(average="macro", num_classes=num_classes)
        self.recall = Recall(average="macro", num_classes=num_classes)
        self.f1_score = F1Score(average="macro", num_classes=num_classes)

    def _logits_to_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        prediction_probabilities = torch.softmax(logits, dim=1)
        _, predicted_classes = torch.max(prediction_probabilities, dim=1)

        return predicted_classes


    def _compute_metrics(
          self, predicted: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Union[int, float]]:

        return {
            'accuracy': self.acc(predicted.cpu(), labels.cpu()).item(),
            'precision': self.precision(predicted.cpu(), labels.cpu()).item(),
            'recall': self.recall(predicted.cpu(), labels.cpu()).item(),
            'f1_score': self.f1_score(predicted.cpu(), labels.cpu()).item()
        }
