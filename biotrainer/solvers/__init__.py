from typing import Optional

from .solver import Solver
from .solver_utils import get_mean_and_confidence_range
from .residues_regression_solver import ResiduesRegressionSolver
from .sequence_regression_solver import SequenceRegressionSolver
from .residue_classification_solver import ResidueClassificationSolver
from .residues_classification_solver import ResiduesClassificationSolver
from .sequence_classification_solver import SequenceClassificationSolver

from .metrics_calculator import MetricsCalculator, SequenceClassificationMetricsCalculator, \
    ResidueClassificationMetricsCalculator, ResiduesClassificationMetricsCalculator, \
    SequenceRegressionMetricsCalculator, ResidueRegressionMetricsCalculator, ResiduesRegressionMetricsCalculator

from ..protocols import Protocol


__SOLVERS = {
    Protocol.residue_to_class: ResidueClassificationSolver,
    Protocol.residues_to_class: ResiduesClassificationSolver,
    Protocol.residues_to_value: ResiduesRegressionSolver,
    Protocol.sequence_to_class: SequenceClassificationSolver,
    Protocol.sequence_to_value: SequenceRegressionSolver,
}

__METRICS_CALCULATORS = {
    Protocol.residue_to_class: ResidueClassificationMetricsCalculator,
    Protocol.residues_to_class: ResiduesClassificationMetricsCalculator,
    Protocol.residues_to_value: ResiduesRegressionMetricsCalculator,
    Protocol.sequence_to_class: SequenceClassificationMetricsCalculator,
    Protocol.sequence_to_value: SequenceRegressionMetricsCalculator,
}


def get_solver(protocol: Protocol, name: str,
               network: Optional = None, optimizer: Optional = None, loss_function: Optional = None,
               device: Optional = None, number_of_epochs: Optional = None,
               patience: Optional = None, epsilon: Optional = None, log_writer: Optional = None,
               log_dir: Optional = None, num_classes: Optional[int] = 0,
               **kwargs
               ) -> Solver:
    solver = __SOLVERS.get(protocol)
    metrics_calc = get_metrics_calculator(protocol=protocol, device=device, num_classes=num_classes)

    if not solver:
        raise NotImplementedError
    else:
        return solver(
            name=name, protocol=protocol,
            network=network, optimizer=optimizer, loss_function=loss_function, metrics_calculator=metrics_calc,
            device=device, number_of_epochs=number_of_epochs,
            patience=patience, epsilon=epsilon, log_writer=log_writer,
            log_dir=log_dir, num_classes=num_classes
        )


def get_metrics_calculator(protocol: Protocol, device: Optional = None, num_classes: Optional[int] = 0):
    metrics_calc = __METRICS_CALCULATORS.get(protocol)

    if not metrics_calc:
        raise NotImplementedError
    else:
        return metrics_calc(device=device, num_classes=num_classes)


__all__ = [
    'Solver',
    'MetricsCalculator',
    'get_solver',
    'get_metrics_calculator',
    'get_mean_and_confidence_range'
]
