import torch

from typing import Optional, Dict, Union

from .Solver import Solver


class SequenceRegressionSolver(Solver):

    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:
        return network_output.flatten().float()

    def _compute_metrics(
          self, predicted: torch.Tensor, labels: torch.Tensor, masks: Optional[torch.BoolTensor] = None
    ) -> Dict[str, Union[int, float]]:

        mse = torch.square((predicted - labels).float()).sum() / len(labels)

        return {
            'mse': mse.item(),
            'rmse': torch.sqrt(mse).item()
        }
