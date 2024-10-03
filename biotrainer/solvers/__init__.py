from typing import Optional
from .solver import Solver
from .solver_utils import get_mean_and_confidence_range
from .residues_regression_solver import ResiduesRegressionSolver
from .sequence_regression_solver import SequenceRegressionSolver
from .residue_classification_solver import ResidueClassificationSolver
from .residues_classification_solver import ResiduesClassificationSolver
from .sequence_classification_solver import SequenceClassificationSolver

from ..protocols import Protocol

__SOLVERS = {
    Protocol.residue_to_class: ResidueClassificationSolver,
    Protocol.residues_to_class: ResiduesClassificationSolver,
    Protocol.residues_to_value: ResiduesRegressionSolver,
    Protocol.sequence_to_class: SequenceClassificationSolver,
    Protocol.sequence_to_value: SequenceRegressionSolver,
}


def get_solver(protocol: Protocol, name: str,
               network: Optional = None, optimizer: Optional = None, loss_function: Optional = None,
               device: Optional = None, number_of_epochs: Optional = None,
               patience: Optional = None, epsilon: Optional = None, log_writer: Optional = None,
               log_dir: Optional = None, num_classes: Optional[int] = 0,
               ):

    solver = __SOLVERS.get(protocol)

    if not solver:
        raise NotImplementedError
    else:
        return solver(
            name=name, protocol=protocol,
            network=network, optimizer=optimizer, loss_function=loss_function,
            device=device, number_of_epochs=number_of_epochs,
            patience=patience, epsilon=epsilon, log_writer=log_writer,
            log_dir=log_dir, num_classes=num_classes
        )


__all__ = [
    'Solver',
    'get_solver',
    'get_mean_and_confidence_range'
]
