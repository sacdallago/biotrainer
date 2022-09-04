import torch

from typing import Dict, Union
from torchmetrics import Accuracy, Precision, Recall, F1Score, SpearmanCorrCoef, MatthewsCorrCoef

from .Solver import Solver


class SequenceClassificationSolver(Solver):

    def __init__(self, *args, **kwargs):
        super(SequenceClassificationSolver, self).__init__(*args, **kwargs)
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

    def _logits_to_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        prediction_probabilities = torch.softmax(logits, dim=1)
        _, predicted_classes = torch.max(prediction_probabilities, dim=1)

        return predicted_classes


    def _compute_metrics(
          self, predicted: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Union[int, float]]:

        precision_per_class = self.precision_per_class(predicted.cpu(), labels.cpu())
        precisions = {'- precission class {}'.format(i): precision_per_class[i] for i in range(self.num_classes)}

        recall_per_class = self.recall_per_class(predicted.cpu(), labels.cpu())
        recalls = {'- recall class {}'.format(i): recall_per_class[i] for i in range(self.num_classes)}

        f1_per_class = self.f1_per_class(predicted.cpu(), labels.cpu())
        f1scores = {'- f1_score class {}'.format(i): f1_per_class[i] for i in range(self.num_classes)}

        return {
            'accuracy': self.acc(predicted.cpu(), labels.cpu()).item(),

            'macro-precision': self.macro_precision(predicted.cpu(), labels.cpu()).item(),
            'micro-precision': self.macro_precision(predicted.cpu(), labels.cpu()).item(),
            **precisions,

            'macro-recall': self.macro_recall(predicted.cpu(), labels.cpu()).item(),
            'micro-recall': self.micro_recall(predicted.cpu(), labels.cpu()).item(),
            **recalls,

            'macro-f1_score': self.macro_f1_score(predicted.cpu(), labels.cpu()).item(),
            'micro-f1_score': self.micro_f1_score(predicted.cpu(), labels.cpu()).item(),
            **f1scores,

            'spearmans-corr-coeff': self.scc(predicted.cpu().type(torch.FloatTensor), labels.cpu().type(torch.FloatTensor)).item(),
            'matthews-corr-coeff': self.mcc(predicted.cpu(), labels.cpu()).item(),
        }
