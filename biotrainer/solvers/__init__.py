from .ResidueSolver import ResidueSolver
from .SequenceSolver import SequenceSolver


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


__all__ = [
    'get_solver'
]
