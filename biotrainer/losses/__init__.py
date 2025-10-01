import torch
import torch.nn as nn

from typing import Optional, Union, Dict, Any, Set

from .masked_mse_loss import MaskedMSELoss

from ..protocols import Protocol
from ..utilities import MASK_AND_LABELS_PAD_VALUE

__LOSSES = {
    Protocol.residue_to_class: {
        'cross_entropy_loss': lambda **kwargs: nn.CrossEntropyLoss(**kwargs, ignore_index=MASK_AND_LABELS_PAD_VALUE)
    },
    Protocol.residue_to_value: {
        'mean_squared_error': lambda **kwargs: MaskedMSELoss(ignore_index=MASK_AND_LABELS_PAD_VALUE)
    },
    Protocol.residues_to_class: {
        'cross_entropy_loss': lambda **kwargs: nn.CrossEntropyLoss(**kwargs, ignore_index=MASK_AND_LABELS_PAD_VALUE)
    },
    Protocol.residues_to_value: {
        'mean_squared_error': lambda **kwargs: nn.MSELoss(**kwargs)
    },
    Protocol.sequence_to_class: {
        'cross_entropy_loss': lambda **kwargs: nn.CrossEntropyLoss(**kwargs, ignore_index=MASK_AND_LABELS_PAD_VALUE)
    },
    Protocol.sequence_to_value: {
        'mean_squared_error': lambda **kwargs: nn.MSELoss(**kwargs)
    },
}


def get_loss(protocol: Protocol, loss_choice: str, device: Union[str, torch.device],
             weight: Optional[torch.Tensor] = None,
             use_class_weights: Optional[bool] = False,
             **kwargs):
    loss = __LOSSES.get(protocol).get(loss_choice)

    if not loss:
        raise NotImplementedError

    if use_class_weights:
        assert weight is not None, "Weight must be provided when using class weights"
        return loss(weight=weight).to(device)
    else:
        return loss().to(device)


def get_available_losses_dict() -> Dict[Protocol, Dict[str, Any]]:
    return dict(__LOSSES)


def get_available_losses_set() -> Set[str]:
    all_losses_by_protocol = [list(protocol.keys()) for protocol in get_available_losses_dict().values()]
    all_losses_list = [loss for loss_list in all_losses_by_protocol for loss in loss_list]
    return set(all_losses_list)


__all__ = [
    "get_loss",
    "get_available_losses_dict",
    "get_available_losses_set"
]
