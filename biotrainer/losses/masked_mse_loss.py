import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss capable of ignoring individual residue/target combinations for residue_to_value protocol """
    def __init__(self, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
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

        # Compute MSE loss on valid positions only
        loss = nn.functional.mse_loss(masked_input, masked_target, reduction=self.reduction)
        return loss
