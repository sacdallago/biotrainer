import torch

from typing import Dict, Union, Optional
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
            self, predicted: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[int, float]]:
        def _compute_metric(metric) -> torch.Tensor:
            # To shorten the code below, this delegate function is used
            return self._compute_metric(metric, predicted=predicted, labels=labels)

        metrics_dict = {'accuracy': _compute_metric(self.acc).item()}

        # Multi-class prediction
        if self.num_classes > 2:
            precision_per_class = _compute_metric(self.precision_per_class)
            precisions = {'- precission class {}'.format(i): precision_per_class[i].item() for i in
                          range(self.num_classes)}
            metrics_dict['macro-precision'] = _compute_metric(self.macro_precision).item()
            metrics_dict['micro-precision'] = _compute_metric(self.micro_precision).item()
            metrics_dict.update(precisions)

            recall_per_class = _compute_metric(self.recall_per_class)
            recalls = {'- recall class {}'.format(i): recall_per_class[i].item() for i in range(self.num_classes)}
            metrics_dict['macro-recall'] = _compute_metric(self.macro_recall).item()
            metrics_dict['micro-recall'] = _compute_metric(self.micro_recall).item()
            metrics_dict.update(recalls)

            f1_per_class = _compute_metric(self.f1_per_class)
            f1scores = {'- f1_score class {}'.format(i): f1_per_class[i].item() for i in range(self.num_classes)}
            metrics_dict['macro-f1_score'] = _compute_metric(self.macro_f1_score).item()
            metrics_dict['micro-f1_score'] = _compute_metric(self.micro_f1_score).item()
            metrics_dict.update(f1scores)
        # Binary prediction
        else:
            metrics_dict['precision'] = _compute_metric(self.macro_precision).item()
            metrics_dict['recall'] = _compute_metric(self.macro_recall).item()
            metrics_dict['f1_score'] = _compute_metric(self.macro_f1_score).item()

        metrics_dict['spearmans-corr-coeff'] = _compute_metric(self.scc).item()
        metrics_dict['matthews-corr-coeff'] = _compute_metric(self.mcc).item()

        return metrics_dict
