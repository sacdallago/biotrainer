import torch
import torch.nn as nn

from typing import Callable


class MaskedRegressionLoss(nn.Module):
    """ Masked Regression Loss capable of ignoring individual residue/target combinations for residue_to_value protocol """

    def __init__(self, loss_function: Callable, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
        self.loss_function = loss_function
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Create mask for valid (non-ignored) positions
        mask = target != self.ignore_index

        if not mask.any():
            # If all targets are masked, return zero loss
            return torch.tensor(0.0, device=input.device, requires_grad=True)

        # Apply mask to both input and target
        masked_input = input[mask]
        masked_target = target[mask]

        # Compute loss on valid positions only
        loss = self.loss_function(masked_input, masked_target, reduction=self.reduction)
        return loss
