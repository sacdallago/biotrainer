from .FNN import FNN
from .CNN import CNN
from .LogReg import LogReg
#from .ConvNeXt import ConvNeXt
from .LightAttention import LightAttention
from .model_params import count_parameters

__MODELS = {
    'residue_to_class': {
        'CNN': CNN,
#        'ConvNeXt': ConvNeXt,
        'FNN': FNN,
        'LogReg': LogReg,
    },
    'residues_to_class': {
        'LightAttention': LightAttention,
    },
    'sequence_to_class': {
        'FNN': FNN,
        'LogReg': LogReg
    },
    'sequence_to_value': {
        'FNN': FNN,
        'LogReg': LogReg
    }
}


def get_model(protocol: str, model_choice: str, n_classes: int, n_features: int):
    model = __MODELS.get(protocol).get(model_choice)

    if not model:
        raise NotImplementedError
    else:
        return model(n_classes=n_classes, n_features=n_features)


def get_all_available_models():
    return dict(__MODELS)


__all__ = [
    "get_model",
    "count_parameters",
]
