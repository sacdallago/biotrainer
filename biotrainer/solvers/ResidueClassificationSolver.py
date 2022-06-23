import torch

from typing import Dict, Union

from .Solver import Solver


class ResidueClassificationSolver(Solver):

    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:

        network_type = type(self.network).__name__
        if network_type in ["FNN", "LogReg"]:
            # (Batch_size x protein_Length x Number_classes) => (B x N x L)
            network_output = network_output.permute(0, 2, 1)

        return network_output

    def _logits_to_predictions(self, logits: torch.Tensor) -> torch.Tensor:

        prediction_probabilities = torch.softmax(logits, dim=1)
        _, predicted_classes = torch.max(prediction_probabilities, dim=1)

        return predicted_classes

    def _compute_metrics(
            self, predicted: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Union[int, float]]:
        # This will flatten everything!
        masks = labels != -100
        masks = masks.to(self.device)

        masked_predicted = torch.masked_select(predicted, masks)
        masked_labels = torch.masked_select(labels, masks)

        return {
            'accuracy': ((masked_predicted == masked_labels).float().sum() / len(masked_labels)).item()
        }
