import torch

from typing import Optional, Dict, Union

from .Solver import Solver


class ResidueClassificationSolver(Solver):

    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:

        network_type = type(self.network).__name__
        if network_type in ["FNN", "LogReg"] :
            network_output = network_output.permute(0, 2, 1)  # (Batch_size x protein_Length x Number_classes) => (B x N x L)

        return network_output

    def _logits_to_predictions(self, logits: torch.Tensor) -> torch.Tensor:

        prediction_probabilities = torch.softmax(logits, dim=1)
        _, predicted_classes = torch.max(prediction_probabilities, dim=1)

        return predicted_classes

    def _compute_metrics(
          self, predicted: torch.Tensor, labels: torch.Tensor, masks: Optional[torch.BoolTensor] = None
    ) -> Dict[str, Union[int, float]]:

        # This will flatten everything!
        masked_predicted = torch.masked_select(predicted, masks)
        masked_labels = torch.masked_select(labels, masks)

        return {
            'accuracy': ((masked_predicted == masked_labels).float().sum() / len(masked_labels)).item()
        }
