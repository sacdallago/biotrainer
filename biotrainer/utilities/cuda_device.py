import torch

from typing import Union


def get_device(device: Union[None, str, torch.device] = None) -> torch.device:
    """Returns what the user specified, or defaults to the GPU,
    with a fallback to CPU if no GPU is available."""
    if isinstance(device, torch.device):
        if device.type == "cuda" and torch.cuda.is_available():
            return device
        elif device.type == "mps" and torch.backends.mps.is_available():
            return device
        else:
            return torch.device("cpu")
    elif isinstance(device, str):
        if "cuda" in device and torch.cuda.is_available():
            return torch.device(device)
        elif "mps" in device and torch.backends.mps.is_available():
            return torch.device(device)
        else:
            return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def is_device_cpu(device: Union[None, str, torch.device] = None) -> bool:
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "cpu"
    if device is str:
        return device == "cpu"
    return False
