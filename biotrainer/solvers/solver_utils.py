import torch

from typing import Tuple


def get_mean_and_confidence_bounds(values: torch.Tensor, dimension: int, confidence_level: float) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the mean and confidence range for the given values. Used for bootstrapping error reporting and
    monte carlo dropout.

    :param values: Predicted values
    :param dimension: Dimension to consider for values tensor
    :param confidence_level: Confidence level for result confidence intervals (0.05 => 95% percentile)
    :return: Tuple: Tensor with mean over values and confidence range for each value
    """
    if not 0 < confidence_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, given: {confidence_level}!")

    values_float = values.float()

    mean = torch.mean(values_float, dim=dimension)

    # Calculate percentiles from actual distribution
    lower_percentile = (confidence_level / 2) * 100
    upper_percentile = (1 - confidence_level / 2) * 100

    lower_bound = torch.quantile(values_float, lower_percentile / 100, dim=dimension)
    upper_bound = torch.quantile(values_float, upper_percentile / 100, dim=dimension)

    return mean, lower_bound, upper_bound
