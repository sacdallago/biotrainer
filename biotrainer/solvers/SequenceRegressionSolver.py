import torch

from typing import Dict, Union, Optional
from torchmetrics import SpearmanCorrCoef

from .Solver import Solver


class SequenceRegressionSolver(Solver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scc = SpearmanCorrCoef()

    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:
        return network_output.flatten().float()

    def _compute_metrics(
            self, predicted: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[int, float]]:
        mse = torch.square((predicted - labels).float()).sum() / len(labels)

        return {
            'mse': mse.item(),
            'rmse': torch.sqrt(mse).item(),
            'spearmans-corr-coeff': self._compute_metric(self.scc, predicted.cpu().type(torch.FloatTensor),
                                                         labels.cpu().type(torch.FloatTensor)).item()
        }
