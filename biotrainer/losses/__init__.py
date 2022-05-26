import torch
import torch.nn as nn

from typing import Optional

__LOSSES = {
    'residue_to_class': {
        'cross_entropy_loss': nn.CrossEntropyLoss
    },
    'sequence_to_class': {
        'cross_entropy_loss': nn.CrossEntropyLoss
    }
}


def get_loss(protocol: str, loss_choice: str, weight: Optional[torch.Tensor] = None):
    loss = __LOSSES.get(protocol).get(loss_choice)

    if not loss:
        raise NotImplementedError
    else:
        return loss(weight=weight)


__all__ = [
    "get_loss"
]
