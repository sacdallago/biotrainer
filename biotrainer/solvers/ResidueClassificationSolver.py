import torch

from typing import Dict, Union
from torchmetrics import Accuracy, Precision, Recall, F1Score, SpearmanCorrCoef, MatthewsCorrCoef

from .Solver import Solver
from ..utilities import MASK_AND_LABELS_PAD_VALUE


class ResidueClassificationSolver(Solver):

    def __init__(self, *args, **kwargs):
        super(ResidueClassificationSolver, self).__init__(*args, **kwargs)
        # Init metrics
        self.num_classes = kwargs['num_classes']
        self.acc = Accuracy(average="micro", num_classes=self.num_classes)

        self.macro_precision = Precision(average="macro", num_classes=self.num_classes)
        self.micro_precision = Precision(average="micro", num_classes=self.num_classes)
        self.precision_per_class = Precision(average="none", num_classes=self.num_classes)

        self.macro_recall = Recall(average="macro", num_classes=self.num_classes)
        self.micro_recall = Recall(average="micro", num_classes=self.num_classes)
        self.recall_per_class = Recall(average="none", num_classes=self.num_classes)
        
        self.macro_f1_score = F1Score(average="macro", num_classes=self.num_classes)
        self.micro_f1_score = F1Score(average="macro", num_classes=self.num_classes)
        self.f1_per_class = F1Score(average="none", num_classes=self.num_classes)

        self.scc = SpearmanCorrCoef()
        self.mcc = MatthewsCorrCoef(num_classes=self.num_classes)

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

        precision_per_class = self.precision_per_class(masked_predicted.cpu(), masked_labels.cpu())
        precisions = {'- precission class {}'.format(i): precision_per_class[i] for i in range(self.num_classes)}

        recall_per_class = self.recall_per_class(masked_predicted.cpu(), masked_labels.cpu())
        recalls = {'- recall class {}'.format(i): recall_per_class[i] for i in range(self.num_classes)}

        f1_per_class = self.f1_per_class(masked_predicted.cpu(), masked_labels.cpu())
        f1scores = {'- f1_score class {}'.format(i): f1_per_class[i] for i in range(self.num_classes)}

        return {
            'accuracy': self.acc(masked_predicted.cpu(), masked_labels.cpu()).item(),

            'macro-precision': self.macro_precision(masked_predicted.cpu(), masked_labels.cpu()).item(),
            'micro-precision': self.macro_precision(masked_predicted.cpu(), masked_labels.cpu()).item(),
            **precisions,

            'macro-recall': self.macro_recall(masked_predicted.cpu(), masked_labels.cpu()).item(),
            'micro-recall': self.micro_recall(masked_predicted.cpu(), masked_labels.cpu()).item(),
            **recalls,

            'macro-f1_score': self.macro_f1_score(masked_predicted.cpu(), masked_labels.cpu()).item(),
            'micro-f1_score': self.micro_f1_score(masked_predicted.cpu(), masked_labels.cpu()).item(),
            **f1scores,

            'spearmans-corr-coeff': self.scc(masked_predicted.cpu().type(torch.FloatTensor), masked_labels.cpu().type(torch.FloatTensor)).item(),
            'matthews-corr-coeff': self.mcc(masked_predicted.cpu(), masked_labels.cpu()).item(),
        }
