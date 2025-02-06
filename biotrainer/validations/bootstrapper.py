from typing import Dict, Any

from ..protocols import Protocol
from ..inference import Inferencer
from ..solvers import MetricsCalculator


class Bootstrapper:
    @staticmethod
    def bootstrap(protocol: Protocol, device, bootstrapping_iterations: int, metrics_calculator: MetricsCalculator,
                  mapped_predictions: Dict[str, Any], test_loader) -> Dict[str, Any]:
        try:
            max_prediction_length = len(max(mapped_predictions.values(), key=len))
        except TypeError:
            max_prediction_length = 1

        all_predictions_dict = {
            idx: Inferencer._pad_tensor(protocol=protocol,
                                        target=pred,
                                        length_to_pad=max_prediction_length,
                                        device=device) for idx, pred in mapped_predictions.items()
        }
        target_dict = {
            idx: Inferencer._pad_tensor(protocol=protocol,
                                        target=target,
                                        length_to_pad=max_prediction_length,
                                        device=device)
            for idx, target in
            zip(test_loader.dataset.ids, test_loader.dataset.targets)}
        seq_ids = list(target_dict.keys())

        sample_size = len(seq_ids)
        confidence_level = 0.05
        bootstrapping_results = Inferencer._do_bootstrapping(iterations=bootstrapping_iterations,
                                                             sample_size=sample_size,
                                                             confidence_level=confidence_level,
                                                             seq_ids=seq_ids,
                                                             all_predictions_dict=all_predictions_dict,
                                                             all_targets_dict=target_dict,
                                                             metrics_calculator=metrics_calculator.reset())
        bootstrapping_dict = {"results": bootstrapping_results, "iterations": bootstrapping_iterations,
                              "sample_size": sample_size, "confidence_level": confidence_level}
        return bootstrapping_dict
