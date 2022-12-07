import torch

from typing import Dict, Union
from torchmetrics import Accuracy, Precision, Recall, F1Score, SpearmanCorrCoef, MatthewsCorrCoef

from .Solver import Solver


class ClassificationSolver(Solver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Init metrics
        self.num_classes = kwargs['num_classes']
        task = "multiclass" if self.num_classes >= 1 else "binary"
        self.acc = Accuracy(task=task, average="micro", num_classes=self.num_classes)

        self.macro_precision = Precision(task=task, average="macro", num_classes=self.num_classes)
        self.micro_precision = Precision(task=task, average="micro", num_classes=self.num_classes)
        self.precision_per_class = Precision(task=task, average="none", num_classes=self.num_classes)

        self.macro_recall = Recall(task=task, average="macro", num_classes=self.num_classes)
        self.micro_recall = Recall(task=task, average="micro", num_classes=self.num_classes)
        self.recall_per_class = Recall(task=task, average="none", num_classes=self.num_classes)

        self.macro_f1_score = F1Score(task=task, average="macro", num_classes=self.num_classes)
        self.micro_f1_score = F1Score(task=task, average="micro", num_classes=self.num_classes)
        self.f1_per_class = F1Score(task=task, average="none", num_classes=self.num_classes)

        self.scc = SpearmanCorrCoef()
        self.mcc = MatthewsCorrCoef(task=task, num_classes=self.num_classes)

    def _compute_metrics(
            self, predicted: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Union[int, float]]:
        metrics_dict = {'accuracy': self.acc(predicted.cpu(), labels.cpu()).item()}

        # Multi-class prediction
        if self.num_classes > 2:
            precision_per_class = self.precision_per_class(predicted.cpu(), labels.cpu())
            precisions = {'- precission class {}'.format(i): precision_per_class[i].item() for i in
                          range(self.num_classes)}
            metrics_dict['macro-precision'] = self.macro_precision(predicted.cpu(), labels.cpu()).item()
            metrics_dict['micro-precision'] = self.micro_precision(predicted.cpu(), labels.cpu()).item()
            metrics_dict.update(precisions)

            recall_per_class = self.recall_per_class(predicted.cpu(), labels.cpu())
            recalls = {'- recall class {}'.format(i): recall_per_class[i].item() for i in range(self.num_classes)}
            metrics_dict['macro-recall'] = self.macro_recall(predicted.cpu(), labels.cpu()).item()
            metrics_dict['micro-recall'] = self.micro_recall(predicted.cpu(), labels.cpu()).item()
            metrics_dict.update(recalls)

            f1_per_class = self.f1_per_class(predicted.cpu(), labels.cpu())
            f1scores = {'- f1_score class {}'.format(i): f1_per_class[i].item() for i in range(self.num_classes)}
            metrics_dict['macro-f1_score'] = self.macro_f1_score(predicted.cpu(), labels.cpu()).item()
            metrics_dict['micro-f1_score'] = self.micro_f1_score(predicted.cpu(), labels.cpu()).item()
            metrics_dict.update(f1scores)
        # Binary prediction
        else:
            metrics_dict['precision'] = self.macro_precision(predicted.cpu(), labels.cpu()).item()
            metrics_dict['recall'] = self.macro_recall(predicted.cpu(), labels.cpu()).item()
            metrics_dict['f1_score'] = self.macro_f1_score(predicted.cpu(), labels.cpu()).item()

        metrics_dict['spearmans-corr-coeff'] = self.scc(predicted.cpu().float(), labels.cpu().float()).item()
        metrics_dict['matthews-corr-coeff'] = self.mcc(predicted.cpu(), labels.cpu()).item()

        return metrics_dict
