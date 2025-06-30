import torch

from torch.utils.data import DataLoader
from contextlib import nullcontext as _nullcontext
from typing import Dict, Union, Optional, Callable, List

from .solver import Solver
from .solver_utils import get_mean_and_confidence_range


class ResidueClassificationSolver(Solver):
    def _transform_network_output(self, network_output: torch.Tensor) -> torch.Tensor:
        network_type = type(self.network.get_downstream_model()).__name__

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
            raise ValueError(f"Confidence level must be between 0 and 1, given: {confidence_level}!")

        mapped_predictions = {}

        for i, (seq_ids, X, y, lengths) in enumerate(dataloader):
            dropout_iterations = self._do_dropout_iterations(X, lengths, n_forward_passes)

            # Get class probabilities individually for each sequence
            class_probabilities_by_sequence = {}
            for dropout_iteration in dropout_iterations:
                dropout_probabilities = dropout_iteration["probabilities"]
                for idx, class_probabilities in enumerate(dropout_probabilities):
                    if seq_ids[idx] not in class_probabilities_by_sequence.keys():
                        class_probabilities_by_sequence[seq_ids[idx]] = []
                    class_probabilities_by_sequence[seq_ids[idx]].append(class_probabilities)

            # Calculate dropout mean and confidence range for each residue in sequence
            seq_idx = 0
            for seq_id, dropout_residues in class_probabilities_by_sequence.items():
                stacked_residues_tensor = torch.stack([torch.tensor(by_class) for by_class in dropout_residues], dim=1)

                dropout_mean, confidence_range = get_mean_and_confidence_range(values=stacked_residues_tensor,
                                                                               dimension=1,
                                                                               confidence_level=confidence_level)
                _, prediction_by_mean = torch.max(dropout_mean, dim=0)

                # Create dict with seq_id: prediction
                mapped_predictions[seq_id] = []
                for residue_idx, residue_prediction in enumerate(prediction_by_mean):
                    mapped_predictions[seq_id].append(
                        {"prediction": residue_prediction.item(),
                         "all_predictions": [dropout_iteration["prediction"][seq_idx][residue_idx] for
                                             dropout_iteration in dropout_iterations],
                         "mcd_mean": dropout_mean.T[residue_idx],
                         "mcd_lower_bound": (
                                 dropout_mean.T[residue_idx] - confidence_range.T[residue_idx]),
                         "mcd_upper_bound": (
                                 dropout_mean.T[residue_idx] + confidence_range.T[residue_idx]),
                         "confidence_range": confidence_range.T[residue_idx],
                         })

                seq_idx += 1
            return {
                'mapped_predictions': mapped_predictions
            }

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
