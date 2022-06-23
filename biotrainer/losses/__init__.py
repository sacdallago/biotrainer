import torch
import torch.nn as nn

from typing import Optional

from ..utilities import MASK_AND_LABELS_PAD_VALUE

__LOSSES = {
    'residue_to_class': {
        'cross_entropy_loss': lambda **kwargs: nn.CrossEntropyLoss(**kwargs, ignore_index=MASK_AND_LABELS_PAD_VALUE)
    },
    'sequence_to_class': {
        'cross_entropy_loss': lambda **kwargs: nn.CrossEntropyLoss(**kwargs, ignore_index=MASK_AND_LABELS_PAD_VALUE)
    },
    'sequence_to_value': {
        'mean_squared_error': lambda **kwargs: nn.MSELoss(**kwargs)
    }
}


def get_loss(protocol: str, loss_choice: str, weight: Optional[torch.Tensor] = None):
    loss = __LOSSES.get(protocol).get(loss_choice)

    if not loss:
        raise NotImplementedError
    else:
        if weight is not None:
            return loss(weight=weight)
        else:
            return loss()


__all__ = [
    "get_loss"
]
