from typing import Optional
from .ResidueClassificationSolver import ResidueClassificationSolver
from .ResiduesClassificationSolver import ResiduesClassificationSolver
from .SequenceClassificationSolver import SequenceClassificationSolver
from .SequenceRegressionSolver import SequenceRegressionSolver
from .Solver import Solver

__SOLVERS = {
    'residue_to_class': ResidueClassificationSolver,
    'residues_to_class': ResiduesClassificationSolver,
    'sequence_to_class': SequenceClassificationSolver,
    'sequence_to_value': SequenceRegressionSolver,
}


def get_solver(protocol: str, name: str,
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
            name=name,
            network=network, optimizer=optimizer, loss_function=loss_function,
            device=device, number_of_epochs=number_of_epochs,
            patience=patience, epsilon=epsilon, log_writer=log_writer,
            log_dir=log_dir, num_classes=num_classes
        )


__all__ = [
    'Solver',
    'get_solver',
]
