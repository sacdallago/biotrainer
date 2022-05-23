from typing import List
from .ResidueSolver import ResidueSolver
from .SequenceSolver import SequenceSolver
from .MetricsCalculator import MetricsCalculator

__SOLVERS = {
    'residue_to_class': ResidueSolver,
    'sequence_to_class': SequenceSolver
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
