import torch

from typing import Union


def get_device(device: Union[None, str, torch.device] = None) -> torch.device:
    """Returns what the user specified, or defaults to the GPU,
    with a fallback to CPU if no GPU is available."""
    if isinstance(device, torch.device):
        return device
    elif device:
        return torch.device(device)
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
