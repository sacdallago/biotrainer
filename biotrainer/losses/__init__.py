import torch
import torch.nn as nn

from typing import Optional, Union

from ..utilities import MASK_AND_LABELS_PAD_VALUE

__LOSSES = {
    'residue_to_class': {
        'cross_entropy_loss': lambda **kwargs: nn.CrossEntropyLoss(**kwargs, ignore_index=MASK_AND_LABELS_PAD_VALUE)
    },
    'residues_to_class': {
        'cross_entropy_loss': lambda **kwargs: nn.CrossEntropyLoss(**kwargs, ignore_index=MASK_AND_LABELS_PAD_VALUE)
    },
    'sequence_to_class': {
        'cross_entropy_loss': lambda **kwargs: nn.CrossEntropyLoss(**kwargs, ignore_index=MASK_AND_LABELS_PAD_VALUE)
    },
    'sequence_to_value': {
        'mean_squared_error': lambda **kwargs: nn.MSELoss(**kwargs)
    },
    'protein_protein_interaction': {
        'cross_entropy_loss': lambda **kwargs: nn.BCEWithLogitsLoss(**kwargs)
    }
}


def get_loss(protocol: str, loss_choice: str, device: Union[str, torch.device],
             weight: Optional[torch.Tensor] = None):
    loss = __LOSSES.get(protocol).get(loss_choice)

    if not loss:
        raise NotImplementedError
    else:
        if weight is not None:
            if "_interaction" in protocol:
                # https://discuss.pytorch.org/t/bcewithlogitsloss-and-class-weights/88837
                return loss(pos_weight=(weight[1] / weight[0])).to(device)
            else:
                return loss(weight=weight).to(device)
        else:
            return loss().to(device)


__all__ = [
    "get_loss"
]
