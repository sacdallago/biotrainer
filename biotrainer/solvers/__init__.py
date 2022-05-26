from typing import List
from .ResidueClassificationSolver import ResidueClassificationSolver
from .SequenceClassificationSolver import SequenceClassificationSolver
from .MetricsCalculator import MetricsCalculator

__SOLVERS = {
    'residue_to_class': ResidueClassificationSolver,
    'sequence_to_class': SequenceClassificationSolver
}


def get_solver(protocol: str, **kwargs):
    solver = __SOLVERS.get(protocol)

    if not solver:
        raise NotImplementedError
    else:
        return solver(**kwargs)


def get_metrics_calculator(protocol: str, metrics_list: List[str]):
    metrics_calculator = MetricsCalculator(protocol, metrics_list)

    return metrics_calculator


__all__ = [
    'get_solver',
    'get_metrics_calculator',
]
