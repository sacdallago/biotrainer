import torch

from typing import Tuple
from scipy.stats import norm


def get_mean_and_confidence_range(values: torch.Tensor, dimension: int, n: int, confidence_level: float) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the mean and confidence range for the given values. Used for error reporting and monte carlo dropout.
    :param values: Predicted values
    :param dimension: Dimension to consider for values tensor
    :param n: Number of samples
    :param confidence_level: Confidence level for result confidence intervals (0.05 => 95% percentile)
    :return: Tuple: Tensor with mean over values and confidence range for each value
    """
    if not 0 < confidence_level < 1:
        raise Exception(f"Confidence level must be between 0 and 1, given: {confidence_level}!")

    std_dev, mean = torch.std_mean(values, dim=dimension, unbiased=True)
    z_score = norm.ppf(q=1 - (confidence_level / 2))
    confidence_range = z_score * std_dev / (n ** 0.5)
    return mean, confidence_range
