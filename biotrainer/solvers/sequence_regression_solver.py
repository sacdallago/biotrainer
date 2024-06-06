import torch

from typing import Dict, Union, Optional
from torchmetrics import SpearmanCorrCoef, MeanSquaredError

from .solver import Solver


class SequenceRegressionSolver(Solver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mse = MeanSquaredError(squared=True)
        self.rmse = MeanSquaredError(squared=False)
        self.scc = SpearmanCorrCoef()

    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:
        return network_output.flatten().float()

    def _compute_metrics(
            self, predicted: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[int, float]]:

        return {
            'mse': self._compute_metric(self.mse, predicted, labels).item(),
            'rmse': self._compute_metric(self.rmse, predicted, labels).item(),
            'spearmans-corr-coeff': self._compute_metric(self.scc, predicted, labels).item()
        }
