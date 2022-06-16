from typing import Optional
from .ResidueClassificationSolver import ResidueClassificationSolver
from .SequenceClassificationSolver import SequenceClassificationSolver
from .SequenceRegressionSolver import SequenceRegressionSolver
from .Solver import Solver

__SOLVERS = {
    'residue_to_class': ResidueClassificationSolver,
    'sequence_to_class': SequenceClassificationSolver,
    'sequence_to_value': SequenceRegressionSolver
}


def get_solver(protocol: str,
               network: Optional = None, optimizer: Optional = None, loss_function: Optional = None,
               device: Optional = None, number_of_epochs: Optional = None,
               patience: Optional = None, epsilon: Optional = None, log_writer: Optional = None,
               experiment_dir: Optional = None
               ):

    solver = __SOLVERS.get(protocol)

    if not solver:
        raise NotImplementedError
    else:
        return solver(
            network=network, optimizer=optimizer, loss_function=loss_function,
            device=device, number_of_epochs=number_of_epochs,
            patience=patience, epsilon=epsilon, log_writer=log_writer,
            experiment_dir=experiment_dir
        )


__all__ = [
    'get_solver',
]
