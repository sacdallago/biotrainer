import torch

__OPTIMIZERS = {
    'residue_to_class': {
        'adam': lambda **kwargs: torch.optim.Adam(**kwargs, amsgrad=True)
    },
    'residues_to_class': {
        'adam': lambda **kwargs: torch.optim.Adam(**kwargs, amsgrad=True)
    },
    'sequence_to_class': {
        'adam': lambda **kwargs: torch.optim.Adam(**kwargs, amsgrad=True)
    },
    'sequence_to_value': {
        'adam': lambda **kwargs: torch.optim.Adam(**kwargs, amsgrad=True)
    },
}


def get_optimizer(protocol: str, optimizer_choice: str, model_parameters: torch.Tensor, learning_rate: float,
                  **kwargs):

    optimizer = __OPTIMIZERS.get(protocol).get(optimizer_choice)

    if not optimizer:
        raise NotImplementedError
    else:
        return optimizer(params=model_parameters, lr=learning_rate)


__all__ = [
    "get_optimizer"
]
