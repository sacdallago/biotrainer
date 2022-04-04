from .FNN import FNN
from .CNN import CNN
from .LogReg import LogReg
from .ConvNeXt import ConvNeXt

_models = {
    'CNN': CNN,
    'ConvNeXt': ConvNeXt,
    'FNN': FNN,
    'LogReg': LogReg,
}


def get_model(model_choice: str, n_classes: int, n_features: int):
    model = _models.get(model_choice)

    if not model:
        raise NotImplementedError
    else:
        return model(n_classes=n_classes, n_features=n_features)


__all__ = [
    "FNN",
    "CNN",
    "LogReg",
    "get_model"
]