import torch
import psutil

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


def is_device_cuda(device: Union[None, str, torch.device] = None) -> bool:
    if device is None:
        return False
    if not torch.cuda.is_available():
        return False
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if device is str:
        return "cuda" in device
    return False


def is_device_mps(device: Union[None, str, torch.device] = None) -> bool:
    if device is None:
        return False
    if not torch.backends.mps.is_available():
        return False
    if isinstance(device, torch.device):
        return device.type == "mps"
    if device is str:
        return "mps" in device
    return False


def get_device_memory(device: Union[None, str, torch.device] = None) -> float:
    """ Returns the amount of memory available for this device in GB. If it was not possible to calculate, 4 GB is
     used as a conservative default and 8 GB for mps """
    gb_factor = (1024 ** 3)
    conservative_default = 4
    default_mps = 8  # TODO [Cross platform] Improve MacOS RAM estimation

    if device is None or isinstance(device, str):
        device = get_device()
    if is_device_cuda(device):
        free_mem, _ = torch.cuda.mem_get_info(device=device)
        return free_mem / gb_factor
    if is_device_cpu(device):
        vm = psutil.virtual_memory()
        return vm.available / gb_factor
    if is_device_mps(device):
        return default_mps
    return conservative_default
