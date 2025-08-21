import torch

from typing import Set, Dict, Any
from ..protocols import Protocol

__OPTIMIZERS = {
    Protocol.residue_to_class: {
        'adam': lambda **kwargs: torch.optim.Adam(**kwargs, amsgrad=True),
        'sgd': lambda **kwargs: torch.optim.SGD(**kwargs, momentum=0.9)
    },
    Protocol.residue_to_value: {
        'adam': lambda **kwargs: torch.optim.Adam(**kwargs, amsgrad=True),
        'sgd': lambda **kwargs: torch.optim.SGD(**kwargs, momentum=0.9)
    },
    Protocol.residues_to_class: {
        'adam': lambda **kwargs: torch.optim.Adam(**kwargs, amsgrad=True),
        'sgd': lambda **kwargs: torch.optim.SGD(**kwargs, momentum=0.9)
    },
    Protocol.residues_to_value: {
        'adam': lambda **kwargs: torch.optim.Adam(**kwargs, amsgrad=True),
        'sgd': lambda **kwargs: torch.optim.SGD(**kwargs, momentum=0.9)
    },
    Protocol.sequence_to_class: {
        'adam': lambda **kwargs: torch.optim.Adam(**kwargs, amsgrad=True),
        'sgd': lambda **kwargs: torch.optim.SGD(**kwargs, momentum=0.9),
    },
    Protocol.sequence_to_value: {
        'adam': lambda **kwargs: torch.optim.Adam(**kwargs, amsgrad=True),
        'sgd': lambda **kwargs: torch.optim.SGD(**kwargs, momentum=0.9)
    },
}


def get_optimizer(protocol: Protocol, optimizer_choice: str, model_parameters: torch.Tensor, learning_rate: float,
                  **kwargs):

    optimizer = __OPTIMIZERS.get(protocol).get(optimizer_choice)

    if not optimizer:
        raise NotImplementedError
    else:
        return optimizer(params=model_parameters, lr=learning_rate)


def get_available_optimizers_dict() -> Dict[Protocol, Dict[str, Any]]:
    return dict(__OPTIMIZERS)


def get_available_optimizers_set() -> Set[str]:
    all_optimizers_by_protocol = [list(protocol.keys()) for protocol in get_available_optimizers_dict().values()]
    all_optimizers_list = [optimizer for optimizer_list in all_optimizers_by_protocol for optimizer in optimizer_list]
    return set(all_optimizers_list)


__all__ = [
    "get_optimizer",
    "get_available_optimizers_dict",
    "get_available_optimizers_set"
]
