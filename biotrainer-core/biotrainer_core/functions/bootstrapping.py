import numpy as np

from typing import Dict, List, Tuple

from ..data_classes import BootstrappedMetric


def metrics_bootstrap(metrics: Dict[str, List[float]],
                      iterations: int,
                      sample_size: int = -1,
                      confidence_level: float = 0.05,
                      seed: int = 42) -> List[BootstrappedMetric]:
    """ Directly bootstrap over a set of pre-calculated metrics. """
    bootstrapped_result = []
    for metric_name, metric_values in metrics.items():
        metric_values_array = np.array(metric_values)  # Necessary for indexing

        if sample_size == -1:
            sample_size = len(metric_values)

        sample_indices = np.random.RandomState(seed).choice(sample_size, size=(iterations, sample_size),
                                                            replace=True)
        iteration_means = np.mean(metric_values_array[sample_indices], axis=1)
        mean, _, lower, upper = get_mean_and_confidence_bounds(values=iteration_means,
                                                               dimension=0,
                                                               confidence_level=confidence_level)
        bt_m = BootstrappedMetric(name=metric_name, mean=float(mean), lower=float(lower), upper=float(upper),
                                  iterations=iterations, sample_size=sample_size, confidence_level=confidence_level)
        bootstrapped_result.append(bt_m)
    return bootstrapped_result


def get_mean_and_confidence_bounds(values: np.ndarray, dimension: int, confidence_level: float) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the mean and confidence range for the given values. Used for bootstrapping error reporting and
    monte carlo dropout.

    :param values: Predicted values
    :param dimension: Dimension to consider for values tensor
    :param confidence_level: Confidence level for result confidence intervals (0.05 => 95% percentile)
    :return: Tuple: Tensor with mean over values, std.dev and confidence range for each value
    """
    if not 0 < confidence_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, given: {confidence_level}!")

    values_float = values.astype(np.float32)

    mean = np.mean(values_float, axis=dimension)
    std = np.std(values_float, axis=dimension)

    # Calculate percentiles from actual distribution
    lower_percentile = (confidence_level / 2) * 100
    upper_percentile = (1 - confidence_level / 2) * 100

    lower_bound = np.percentile(values_float, lower_percentile, axis=dimension)
    upper_bound = np.percentile(values_float, upper_percentile, axis=dimension)

    return mean, std, lower_bound, upper_bound
