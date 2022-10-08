import torch

from typing import Dict, Union, Optional, Callable
from contextlib import nullcontext as _nullcontext

from .Solver import Solver
from .ClassificationSolver import ClassificationSolver

from ..utilities import MASK_AND_LABELS_PAD_VALUE


class ResidueClassificationSolver(ClassificationSolver, Solver):
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
        masks = labels != MASK_AND_LABELS_PAD_VALUE
        masks = masks.to(self.device)

        masked_predicted = torch.masked_select(predicted, masks)
        masked_labels = torch.masked_select(labels, masks)

        return super()._compute_metrics(predicted=masked_predicted, labels=masked_labels)

    # Gets overwritten to shorten prediction lengths if necessary
    def _training_iteration(
            self, x: torch.Tensor, y: torch.Tensor, step=1, context: Optional[Callable] = None,
            lengths: Optional[torch.LongTensor] = None
    ) -> Dict[str, Union[float, list, Dict[str, Union[float, int]]]]:
        result_dict = super()._training_iteration(x, y, step, context, lengths)
        if not context:
            context = _nullcontext

        with context():
            prediction = result_dict['prediction']
            # If lengths is defined, we need to shorten the residue predictions to the length
            if lengths is not None:
                return_pred = list()
                for pred_x, length_x in zip(prediction, lengths):
                    return_pred.append(pred_x[:length_x])

                result_dict['prediction'] = return_pred

            return result_dict

    # Gets overwritten to shorten prediction lengths if necessary
    def _prediction_iteration(self, x: torch.Tensor, lengths: Optional[torch.LongTensor] = None):
        prediction = super()._prediction_iteration(x, lengths)
        with torch.no_grad():
            if lengths is not None:
                return_pred = list()
                for pred_x, length_x in zip(prediction, lengths):
                    return_pred.append(pred_x[:length_x])

            return return_pred
