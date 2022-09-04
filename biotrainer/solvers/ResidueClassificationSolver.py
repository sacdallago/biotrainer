import torch

from typing import Dict, Union
from torchmetrics import Accuracy, Precision, Recall, F1Score

from .Solver import Solver
from ..utilities import MASK_AND_LABELS_PAD_VALUE


class ResidueClassificationSolver(Solver):

    def __init__(self, *args, **kwargs):
        super(ResidueClassificationSolver, self).__init__(*args, **kwargs)
        # Init metrics
        num_classes = kwargs['num_classes']
        self.acc = Accuracy(average="micro", num_classes=num_classes)
        self.precision = Precision(average="macro", num_classes=num_classes)
        self.recall = Recall(average="macro", num_classes=num_classes)
        self.f1_score = F1Score(average="macro", num_classes=num_classes)

    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:

        network_type = type(self.network).__name__
        if network_type in ["FNN", "LogReg"]:
            # (Batch_size x protein_Length x Number_classes) => (B x N x L)
            network_output = network_output.permute(0, 2, 1)

        return network_output

    def _logits_to_predictions(self, logits: torch.Tensor) -> torch.Tensor:

        prediction_probabilities = torch.softmax(logits, dim=1)
        _, predicted_classes = torch.max(prediction_probabilities, dim=1)

        return predicted_classes

    def _compute_metrics(
            self, predicted: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Union[int, float]]:
        # This will flatten everything!
        masks = labels != MASK_AND_LABELS_PAD_VALUE
        masks = masks.to(self.device)

        masked_predicted = torch.masked_select(predicted, masks)
        masked_labels = torch.masked_select(labels, masks)

        return {
            'accuracy': self.acc(masked_predicted.cpu(), masked_labels.cpu()).item(),
            'precision': self.precision(masked_predicted.cpu(), masked_labels.cpu()).item(),
            'recall': self.recall(masked_predicted.cpu(), masked_labels.cpu()).item(),
            'f1_score': self.f1_score(masked_predicted.cpu(), masked_labels.cpu()).item()
        }
