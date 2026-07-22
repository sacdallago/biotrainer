from biotrainer_core.data_classes import Protocol

from .contact_maps import evaluate_contact_map, compute_contact_precision, evaluate_contact_dataset
from .ndcg import NDCG
from .metrics_calculator import MetricsCalculator, SimpleTorchMetricsCalculator, SimpleCustomMetricsCalculator, \
    ResidueClassificationMetricsCalculator, \
    ResidueRegressionMetricsCalculator, ResiduesClassificationMetricsCalculator, ResiduesRegressionMetricsCalculator, \
    SequenceClassificationMetricsCalculator, SequenceRegressionMetricsCalculator

METRIC_CALCULATORS = {
    Protocol.residue_to_class: ResidueClassificationMetricsCalculator,
    Protocol.residue_to_value: ResidueRegressionMetricsCalculator,
    Protocol.residues_to_class: ResiduesClassificationMetricsCalculator,
    Protocol.residues_to_value: ResiduesRegressionMetricsCalculator,
    Protocol.sequence_to_class: SequenceClassificationMetricsCalculator,
    Protocol.sequence_to_value: SequenceRegressionMetricsCalculator,
}

__all__ = [
    "METRIC_CALCULATORS",
    "NDCG",
    "MetricsCalculator",
    "SimpleTorchMetricsCalculator",
    "SimpleCustomMetricsCalculator",
    "evaluate_contact_map",
    "compute_contact_precision",
    "evaluate_contact_dataset",
]
