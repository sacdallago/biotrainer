import torch

from typing import Dict, Union, Optional
from torchmetrics import SpearmanCorrCoef, MeanSquaredError

from .solver import Solver


class SequenceRegressionSolver(Solver):
    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:
        return network_output.flatten().float()
