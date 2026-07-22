import torch
import numpy as np

from typing import Dict, List, Union, Any
from biotrainer_core.utils.constants import MASK_AND_LABELS_PAD_VALUE
from biotrainer_core.data_classes import Protocol, BiotrainerPrediction, BootstrappedMetric
from biotrainer_core.functions.bootstrapping import get_mean_and_confidence_bounds

from .metrics import MetricsCalculator


class Bootstrapper:

    @staticmethod
    def bootstrap(protocol: Protocol, device,
                  bootstrapping_iterations: int,
                  metrics_calculator: MetricsCalculator,
                  predictions: List[BiotrainerPrediction],
                  test_loader) -> List[BootstrappedMetric]:
        try:
            max_prediction_length = max([len(pred.prediction) for pred in predictions])
        except TypeError:
            max_prediction_length = 1

        all_predictions_dict = {
            pred.seq_id: Bootstrapper._pad_tensor(protocol=protocol,
                                                  target=pred.prediction,
                                                  length_to_pad=max_prediction_length,
                                                  device=device) for pred in predictions
        }
        target_dict = {
            idx: Bootstrapper._pad_tensor(protocol=protocol,
                                          target=target,
                                          length_to_pad=max_prediction_length,
                                          device=device)
            for idx, target in
            zip(test_loader.dataset.ids, test_loader.dataset.targets)}
        seq_ids = list(target_dict.keys())

        sample_size = len(seq_ids)
        confidence_level = 0.05
        return Bootstrapper._do_bootstrapping(iterations=bootstrapping_iterations,
                                              sample_size=sample_size,
                                              confidence_level=confidence_level,
                                              seq_ids=seq_ids,
                                              all_predictions_dict=all_predictions_dict,
                                              all_targets_dict=target_dict,
                                              metrics_calculator=metrics_calculator.reset())

    @staticmethod
    def _do_bootstrapping(iterations: int,
                          sample_size: int,
                          confidence_level: float,
                          seq_ids: List[str],
                          all_predictions_dict: Dict,
                          all_targets_dict: Dict,
                          metrics_calculator: MetricsCalculator) -> List[BootstrappedMetric]:
        """

        :param iterations: Number of iterations to perform bootstrapping
        :param sample_size: Sample size to use for bootstrapping. -1 defaults to all embeddings which is recommended.
                            It is possible, but not recommended to use a sample size larger or smaller
                            than the number of embeddings, because this might render the variance estimate unreliable.
                            See: https://math.mit.edu/~dav/05.dir/class24-prep-a.pdf (6.2)
        :param confidence_level: Confidence level for result error intervals (0.05 => 95% percentile)
        :param seq_ids: List of sequence IDs
        :param all_predictions_dict: Dictionary of all predictions
        :param all_targets_dict: Dictionary of all targets
        :param metrics_calculator: Metrics calculator object
        :return:
        """
        if sample_size == -1:
            sample_size = len(seq_ids)

        # Convert dictionaries to tensors
        all_predictions = torch.stack([all_predictions_dict[seq_id] for seq_id in seq_ids])
        all_targets = torch.stack([all_targets_dict[seq_id] for seq_id in seq_ids])

        # Set random seed
        seed = np.random.get_state()[1][0] if np.random.get_state() else 42
        rng = np.random.RandomState(seed)

        # Generate all random indices at once
        all_indices = rng.choice(len(seq_ids), size=(iterations, sample_size), replace=True)

        iteration_results = []
        for indices in all_indices:
            # Use integer indexing instead of string keys
            sampled_predictions = all_predictions[indices]
            sampled_targets = all_targets[indices]

            iteration_result = metrics_calculator.compute_metrics(
                predicted=sampled_predictions,
                labels=sampled_targets
            )
            iteration_results.append(iteration_result)

        # Process results
        metrics = list(iteration_results[0].keys())
        results = []
        for metric in metrics:
            all_metric_values = torch.tensor([res[metric] for res in iteration_results], dtype=torch.float32)
            mean, _, lower_bound, upper_bound = get_mean_and_confidence_bounds(
                values=all_metric_values.numpy(),
                dimension=0,
                confidence_level=confidence_level
            )
            results.append(BootstrappedMetric(name=metric, mean=mean.item(), lower=lower_bound.item(),
                                              upper=upper_bound.item(), iterations=iterations, sample_size=sample_size,
                                              confidence_level=confidence_level))

        return results

    @staticmethod
    def _pad_tensor(protocol: Protocol, target: Union[Any, torch.Tensor], length_to_pad: int, device):
        target_tensor = torch.as_tensor(target, device=device)
        if protocol in Protocol.per_residue_protocols():
            if target_tensor.shape[0] < length_to_pad:
                padding_size = length_to_pad - target_tensor.shape[0]
                padding = torch.full((padding_size,), MASK_AND_LABELS_PAD_VALUE, dtype=target_tensor.dtype,
                                     device=device)
                return torch.cat([target_tensor, padding])
            else:
                return target_tensor
        else:
            return target_tensor
