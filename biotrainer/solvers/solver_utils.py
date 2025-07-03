import torch

from typing import Tuple
from scipy.stats import norm


def get_mean_and_confidence_range(values: torch.Tensor, dimension: int, confidence_level: float) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the mean and confidence range for the given values. Used for error reporting and monte carlo dropout.
    :param values: Predicted values
    :param dimension: Dimension to consider for values tensor
    :param confidence_level: Confidence level for result confidence intervals (0.05 => 95% percentile)
    :return: Tuple: Tensor with mean over values and confidence range for each value
    """
    if not 0 < confidence_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, given: {confidence_level}!")

    std_dev, mean = torch.std_mean(values, dim=dimension, unbiased=True)
    # Use normal distribution for critical value (z_score)
    z_score = norm.ppf(q=1 - (confidence_level / 2))
    # Confidence range does not include number of iterations:
    # https://moderndive.com/8-confidence-intervals.html#se-method
    # Note that the number of iterations influences the precision of the standard deviation, however.
    confidence_range = z_score * std_dev
    return mean, confidence_range
