import torch
import inspect

from torch import nn
from typing import Dict, Set, Any, Optional

from .cnn import CNN
from .log_reg import LogReg
from .fnn import FNN, DeeperFNN
from .model_params import count_parameters
from .light_attention import LightAttention
from .biotrainer_model import BiotrainerModel
from .fine_tuning_model import FineTuningModel

from ..protocols import Protocol
from ..utilities import get_logger

logger = get_logger(__name__)

__MODELS = {
    Protocol.residue_to_class: {
        'CNN': CNN,
        'FNN': FNN,
        'DeeperFNN': DeeperFNN,
        'LogReg': LogReg,
    },
    Protocol.residue_to_value: {
        'CNN': CNN,
        'FNN': FNN,
        'DeeperFNN': DeeperFNN,
        'LogReg': LogReg,
    },
    Protocol.residues_to_class: {
        'LightAttention': LightAttention,
    },
    Protocol.residues_to_value: {
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

def _initialize_weights(module, init_type='default'):
    """
    Initialize weights for neural network modules

    Args:
        module: PyTorch module
        init_type: Type of initialization ('default', 'normal', 'xavier', 'kaiming')
    """
    if init_type == 'default':
        return

    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        if init_type == 'normal':
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif init_type == 'xavier':
            nn.init.xavier_uniform_(module.weight)
        elif init_type == 'kaiming':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        # Default uses PyTorch's default initialization

        if init_type != 'default':
            nn.init.zeros_(module.bias)


def get_model(protocol: Protocol, model_choice: str, n_classes: int, n_features: int,
              disable_pytorch_compile: Optional[bool] = True,
              model_weights_init: Optional[str] = 'default',
              **kwargs):
    model_class = __MODELS.get(protocol).get(model_choice)
    if not model_class:
        raise NotImplementedError
    else:
        if "dropout_rate" in kwargs.keys() and "dropout_rate" not in inspect.signature(model_class).parameters:
            logger.warning(f"dropout_rate not implemented for model_choice {model_choice}")

        model = model_class(n_classes=n_classes, n_features=n_features, **kwargs)

        # Apply weight initialization if specified
        if model_weights_init != 'default':
            model.apply(lambda m: _initialize_weights(m,
                                                     init_type=model_weights_init,
                                                     ))

        # Disable option for backwards compatibility with older models or if there emerge problems during training
        if not disable_pytorch_compile:
            logger.info(f"Using pytorch model compile mode!")
            model.compile()
        return model


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
    "BiotrainerModel",
    "FineTuningModel",
]
