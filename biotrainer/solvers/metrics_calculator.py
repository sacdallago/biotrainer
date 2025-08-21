from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from typing import Optional, Dict, Union
from torchmetrics import Accuracy, Precision, Recall, F1Score, SpearmanCorrCoef, MatthewsCorrCoef, Metric, \
    MeanSquaredError

from ..utilities import MASK_AND_LABELS_PAD_VALUE


class MetricsCalculator(ABC):
    def __init__(self, device, n_classes: int):
        self.device = device
        self.n_classes = n_classes

    def reset(self) -> MetricsCalculator:
        # Reset all metric attributes that are instances of torchmetrics.Metric
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Metric):
                attr.reset()
        return self

    @abstractmethod
    def compute_metrics(
            self, predicted: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None) -> Dict[str, Union[int, float]]:
        raise NotImplementedError

    @staticmethod
    def _compute_metric(metric: Metric, predicted: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Utility function to calculate metrics either on a per-epoch or a per-batch basis

        :param metric: torchmetrics object
        :param predicted: The predicted label/value for each sample
        :param labels: The actual label for each sample

        :return: metric result calculated via metric object
        """
        if predicted is None and labels is None:
            # Per epoch
            metric_result = metric.compute()
            metric.reset()
            return metric_result
        else:
            # Per batch
            if metric.__class__ == SpearmanCorrCoef:
                # SCC only accepts float tensors
                return metric(predicted.cpu().float(), labels.cpu().float())
            return metric(predicted.cpu(), labels.cpu())


class ClassificationMetricsCalculator(MetricsCalculator):
    def __init__(self, device, n_classes: int):
        super().__init__(device, n_classes)

        task = "multiclass" if self.n_classes > 2 else "binary"

        self.acc = Accuracy(task=task, average="micro", num_classes=self.n_classes)

        self.macro_precision = Precision(task=task, average="macro", num_classes=self.n_classes)
        self.micro_precision = Precision(task=task, average="micro", num_classes=self.n_classes)
        self.precision_per_class = Precision(task=task, average="none", num_classes=self.n_classes)

        self.macro_recall = Recall(task=task, average="macro", num_classes=self.n_classes)
        self.micro_recall = Recall(task=task, average="micro", num_classes=self.n_classes)
        self.recall_per_class = Recall(task=task, average="none", num_classes=self.n_classes)

        self.macro_f1_score = F1Score(task=task, average="macro", num_classes=self.n_classes)
        self.micro_f1_score = F1Score(task=task, average="micro", num_classes=self.n_classes)
        self.f1_per_class = F1Score(task=task, average="none", num_classes=self.n_classes)

        self.scc = SpearmanCorrCoef()
        self.mcc = MatthewsCorrCoef(task=task, num_classes=self.n_classes)

    def compute_metrics(
            self, predicted: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None) -> Dict[str, Union[int, float]]:
        def _compute_metric(metric) -> torch.Tensor:
            # To shorten the code below, this delegate function is used
            return self._compute_metric(metric, predicted=predicted, labels=labels)

        metrics_dict = {'accuracy': _compute_metric(self.acc).item()}

        # Multi-class prediction
        if self.n_classes > 2:
            precision_per_class = _compute_metric(self.precision_per_class)
            precisions = {'- precision class {}'.format(i): precision_per_class[i].item() for i in
                          range(self.n_classes)}
            metrics_dict['macro-precision'] = _compute_metric(self.macro_precision).item()
            metrics_dict['micro-precision'] = _compute_metric(self.micro_precision).item()
            metrics_dict.update(precisions)

            recall_per_class = _compute_metric(self.recall_per_class)
            recalls = {'- recall class {}'.format(i): recall_per_class[i].item() for i in range(self.n_classes)}
            metrics_dict['macro-recall'] = _compute_metric(self.macro_recall).item()
            metrics_dict['micro-recall'] = _compute_metric(self.micro_recall).item()
            metrics_dict.update(recalls)

            f1_per_class = _compute_metric(self.f1_per_class)
            f1scores = {'- f1_score class {}'.format(i): f1_per_class[i].item() for i in range(self.n_classes)}
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


class RegressionMetricsCalculator(MetricsCalculator):
    def __init__(self, device, n_classes: int):
        super().__init__(device, n_classes)

        self.mse = MeanSquaredError(squared=True)
        self.rmse = MeanSquaredError(squared=False)
        self.scc = SpearmanCorrCoef()

    def compute_metrics(
            self, predicted: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None) -> Dict[str, Union[int, float]]:
        return {
            'mse': self._compute_metric(self.mse, predicted, labels).item(),
            'rmse': self._compute_metric(self.rmse, predicted, labels).item(),
            'spearmans-corr-coeff': self._compute_metric(self.scc, predicted, labels).item()
        }


class SequenceClassificationMetricsCalculator(ClassificationMetricsCalculator):
    pass


class ResidueClassificationMetricsCalculator(ClassificationMetricsCalculator):
    def compute_metrics(
            self, predicted: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None) -> Dict[str, Union[int, float]]:
        if predicted is not None and labels is not None:
            # This will flatten everything!
            masks = labels != MASK_AND_LABELS_PAD_VALUE
            masks = masks.to(self.device)

            masked_predicted = torch.masked_select(predicted, masks)
            masked_labels = torch.masked_select(labels, masks)

            return super().compute_metrics(predicted=masked_predicted, labels=masked_labels)
        else:
            return super().compute_metrics(predicted=predicted, labels=labels)

class ResiduesClassificationMetricsCalculator(ClassificationMetricsCalculator):
    pass

class SequenceRegressionMetricsCalculator(RegressionMetricsCalculator):
    pass

class ResidueRegressionMetricsCalculator(RegressionMetricsCalculator):
    def compute_metrics(
            self, predicted: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None) -> Dict[str, Union[int, float]]:
        predicted_flattened = predicted.flatten() if predicted is not None else None
        labels_flattened = labels.flatten() if labels is not None else None
        return {
            'mse': self._compute_metric(self.mse, predicted, labels).item(),
            'rmse': self._compute_metric(self.rmse, predicted, labels).item(),
            'spearmans-corr-coeff': self._compute_metric(self.scc, predicted_flattened, labels_flattened).item()
        }


class ResiduesRegressionMetricsCalculator(RegressionMetricsCalculator):
    pass
