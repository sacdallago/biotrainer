import torch

from sklearn import metrics
from typing import List, Dict, Any


def _residue_to_class_accuracy(y, y_hat):
    # Flatten and compute numbers for later use
    flat_y = y.flatten()
    total_y = len(y)

    # TODO: The value of the mask should be optionable!!!!
    total_to_consider = int(torch.sum(y == -100))
    flat_predicted_classes = y_hat.flatten().cpu()

    # Count how many match
    unmasked_accuracy = metrics.accuracy_score(flat_y, flat_predicted_classes, normalize=False)
    accuracy = (unmasked_accuracy - total_to_consider) / (total_y - total_to_consider)
    return accuracy


_METRICS_BY_PROTOCOL = {
    "residue_to_class": {
        "accuracy": _residue_to_class_accuracy
    },
    "residues_to_class": {
        "accuracy": lambda y, y_hat: metrics.accuracy_score(y.cpu(), y_hat.cpu(), normalize=True)
    },
    "sequence_to_class": {
        "accuracy": lambda y, y_hat: metrics.accuracy_score(y.cpu(), y_hat.cpu(), normalize=True)
    },
}


class MetricsCalculator:

    def __init__(self, protocol: str, metric_list: List[str]):
        self._protocol = protocol
        self._metric_algorithms = list()
        for metric in metric_list:
            self._metric_algorithms.append((metric, self._get_metric_by_protocol(metric)))

    def _get_metric_by_protocol(self, metric: str):
        metric_function = _METRICS_BY_PROTOCOL.get(self._protocol).get(metric)

        if not metric_function:
            raise NotImplementedError

        return metric_function

    def calculate_metrics(self, y, y_hat) -> Dict[str, Any]:
        metrics_dict = dict()
        for metric_name, metric_algo in self._metric_algorithms:
            metrics_dict[metric_name] = metric_algo(y, y_hat)
        return metrics_dict
