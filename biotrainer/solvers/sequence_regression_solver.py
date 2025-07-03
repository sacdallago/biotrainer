import torch

from .solver import Solver


class SequenceRegressionSolver(Solver):
    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:
        return network_output.flatten().float()
