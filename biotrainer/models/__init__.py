import torch
import inspect
import logging

from typing import Dict, Set, Any

from .CNN import CNN
from .LogReg import LogReg
from .FNN import FNN, DeeperFNN
from .LightAttention import LightAttention
from .model_params import count_parameters

from ..protocols import Protocol

logger = logging.getLogger(__name__)

__MODELS = {
    Protocol.residue_to_class: {
        'CNN': CNN,
        'FNN': FNN,
        'DeeperFNN': DeeperFNN,
        'LogReg': LogReg,
    },
    Protocol.residues_to_class: {
        'LightAttention': LightAttention,
    },
    Protocol.sequence_to_class: {
        'FNN': FNN,
        'DeeperFNN': DeeperFNN,
        'LogReg': LogReg
    },
    Protocol.sequence_to_value: {
        'FNN': FNN,
        'DeeperFNN': DeeperFNN,
        'LogReg': LogReg
    },
}


def get_model(protocol: Protocol, model_choice: str, n_classes: int, n_features: int,
              **kwargs):
    model_class = __MODELS.get(protocol).get(model_choice)

    if not model_class:
        raise NotImplementedError
    else:
        if "dropout_rate" in kwargs.keys() and "dropout_rate" not in inspect.signature(model_class).parameters:
            logger.warning(f"dropout_rate not implemented for model_choice {model_choice}")

        model = model_class(n_classes=n_classes, n_features=n_features, **kwargs)
        return torch.compile(model, disable=True)  # Disabled - compile() does not seem to be fully production-ready


def get_available_models_dict() -> Dict[Protocol, Dict[str, Any]]:
    return dict(__MODELS)


def get_available_models_set() -> Set[str]:
    all_models_by_protocol = [list(protocol.keys()) for protocol in get_available_models_dict().values()]
    all_models_list = [model for model_list in all_models_by_protocol for model in model_list]
    return set(all_models_list)


__all__ = [
    "get_model",
    "get_available_models_dict",
    "get_available_models_set",
    "count_parameters",
]
