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

            sequence_probabilities = {}
            for dropout_iteration in dropout_iterations:
                dropout_probabilities = dropout_iteration["probabilities"]
                for idx, sequence in enumerate(dropout_probabilities):
                    if seq_ids[idx] not in sequence_probabilities.keys():
                        sequence_probabilities[seq_ids[idx]] = []
                    sequence_probabilities[seq_ids[idx]].append(sequence)

            for seq_id, dropout_residues in sequence_probabilities.items():
                stacked_residues_tensor = torch.stack([torch.tensor(by_class) for by_class in dropout_residues], dim=1)

                dropout_mean, confidence_range = get_mean_and_confidence_range(values=stacked_residues_tensor,
                                                                               dimension=1,
                                                                               n=n_forward_passes,
                                                                               confidence_level=confidence_level)
                _, prediction_by_mean = torch.max(dropout_mean, dim=0)

                # Create dict with seq_id: prediction
                mapped_predictions[seq_id] = []
                for residue_idx, residue_prediction in enumerate(prediction_by_mean):
                    mapped_predictions[seq_id].append(
                        {"prediction": residue_prediction.item(),
                         "mcd_mean": dropout_mean.T[residue_idx],
                         "mcd_lower_bound": (
                                 dropout_mean.T[residue_idx] - confidence_range.T[residue_idx]),
                         "mcd_upper_bound": (
                                 dropout_mean.T[residue_idx] + confidence_range.T[residue_idx])
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
            predictions = result_dict['prediction']
            probabilities = result_dict['probabilities']
            # If lengths is defined, we need to shorten the residue predictions and probabilities to the length
            if lengths is not None:
                shortened_predictions = []
                shortened_probabilities = []
                for original_prediction, original_probability, length_to_shorten in zip(predictions, probabilities,
                                                                                        lengths):
                    shortened_predictions.append(original_prediction[:length_to_shorten])
                    shortened_per_class_probabilities = []
                    for per_class_probabilities in original_probability:
                        shortened_per_class_probabilities.append(per_class_probabilities[:length_to_shorten])
                    shortened_probabilities.append(shortened_per_class_probabilities)

                result_dict['prediction'] = shortened_predictions
                result_dict['probabilities'] = shortened_probabilities

            return result_dict
