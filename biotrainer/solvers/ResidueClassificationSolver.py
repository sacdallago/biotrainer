import torch

from scipy.stats import norm
from torch.utils.data import DataLoader
from contextlib import nullcontext as _nullcontext
from typing import Dict, Union, Optional, Callable, List

from .Solver import Solver
from .ClassificationSolver import ClassificationSolver
from .solver_utils import get_mean_and_confidence_range

from ..utilities import MASK_AND_LABELS_PAD_VALUE


class ResidueClassificationSolver(ClassificationSolver, Solver):
    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:
        network_type = type(self.network).__name__
        if network_type in ["FNN", "DeeperFNN", "LogReg"]:
            # (Batch_size x protein_Length x Number_classes) => (B x N x L)
            network_output = network_output.permute(0, 2, 1)

        return network_output

    def _logits_to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=1)

    def _probabilities_to_predictions(self, probabilities: torch.Tensor) -> torch.Tensor:
        _, predicted_classes = torch.max(probabilities, dim=1)

        return predicted_classes

    def inference_monte_carlo_dropout(self, dataloader: DataLoader,
                                      n_forward_passes: int = 30,
                                      confidence_level: float = 0.05):
        """
        Calculate inference results from existing models for given embeddings.
        Adaption needed for residue_to_x tasks because of multiple predictions for each sequence

            dataloader: Dataloader with embeddings
            n_forward_passes: Times to repeat calculation with different dropout nodes enabled
            confidence_level: Confidence level for result confidence intervals (0.05 => 95% percentile)
        """
        if not 0 < confidence_level < 1:
            raise Exception(f"Confidence level must be between 0 and 1, given: {confidence_level}!")

        def enable_dropout(model):
            """ Function to enable the dropout layers during test-time """
            number_dropout_layers = 0
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
                    if m.p > 0.0:
                        number_dropout_layers += 1
            if not number_dropout_layers > 0:
                raise Exception("Trying to do monte carlo dropout inference on model without dropout!")

        mapped_predictions = dict()

        for i, (seq_ids, X, y, lengths) in enumerate(dataloader):
            dropout_iterations = list()
            for idx_forward_pass in range(n_forward_passes):
                self.network = self.network.eval()
                enable_dropout(self.network)
                dropout_iteration_result = self._prediction_iteration(x=X, lengths=lengths)
                dropout_iterations.append(dropout_iteration_result)

            dropout_raw_values = torch.stack([dropout_iteration["probabilities"]
                                              for dropout_iteration in dropout_iterations], dim=1)

            dropout_mean, confidence_range = get_mean_and_confidence_range(values=dropout_raw_values,
                                                                           dimension=1,
                                                                           n=n_forward_passes,
                                                                           confidence_level=confidence_level)
            _, prediction_by_mean = torch.max(dropout_mean, dim=1)

            dropout_mean = dropout_mean.permute(0, 2, 1)
            confidence_range = confidence_range.permute(0, 2, 1)
            # Create dict with seq_id: prediction
            for idx, prediction in enumerate(prediction_by_mean):
                mapped_predictions[seq_ids[idx]] = []
                for residue_idx in range(len(prediction)):
                    mapped_predictions[seq_ids[idx]].append(
                        {"prediction": prediction_by_mean[idx][residue_idx].item(),
                         "mcd_mean": dropout_mean[idx][residue_idx],
                         "mcd_lower_bound": (
                                 dropout_mean[idx][residue_idx] - confidence_range[idx][residue_idx]),
                         "mcd_upper_bound": (
                                 dropout_mean[idx][residue_idx] + confidence_range[idx][residue_idx])
                         })

        return {
            'mapped_predictions': mapped_predictions
        }

    def _compute_metrics(
            self, predicted: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[int, float]]:
        if predicted is not None and labels is not None:
            # This will flatten everything!
            masks = labels != MASK_AND_LABELS_PAD_VALUE
            masks = masks.to(self.device)

            masked_predicted = torch.masked_select(predicted, masks)
            masked_labels = torch.masked_select(labels, masks)

            return super()._compute_metrics(predicted=masked_predicted, labels=masked_labels)
        else:
            return super()._compute_metrics(predicted=predicted, labels=labels)

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
    def _prediction_iteration(self, x: torch.Tensor, lengths: Optional[torch.LongTensor] = None) -> Dict[str, List]:
        result_dict = super()._prediction_iteration(x, lengths)
        with torch.no_grad():
            prediction = result_dict['prediction']
            # If lengths is defined, we need to shorten the residue predictions to the length
            if lengths is not None:
                return_pred = list()
                for pred_x, length_x in zip(prediction, lengths):
                    return_pred.append(pred_x[:length_x])

                result_dict['prediction'] = return_pred

            return result_dict
