import torch
import psutil

from typing import Union


def get_device(device: Union[None, str, torch.device] = None) -> torch.device:
    """Returns what the user specified, or auto-selects the best available device.

    Auto-selection (device=None) falls back to CPU when no accelerator is present. An explicitly requested device
    is never silently downgraded - a run that was asked for a GPU should fail loudly rather than spend hours on
    the CPU."""
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if isinstance(device, str):
        try:
            device = torch.device(device)
        except RuntimeError as e:
            raise ValueError(f"Device '{device}' could not be understood: {e}") from e

    if device.type == "cpu":
        return device
    if device.type == "cuda" and torch.cuda.is_available():
        return device
    if device.type == "mps" and torch.backends.mps.is_available():
        return device
    raise ValueError(f"Device '{device}' was requested, but it is not available on this machine "
                     f"(cuda: {torch.cuda.is_available()}, mps: {torch.backends.mps.is_available()}). "
                     f"Pass device=None to select the best available device, or 'cpu' to force it.")


def is_device_cpu(device: Union[None, str, torch.device] = None) -> bool:
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "cpu"
    if isinstance(device, str):
        return device == "cpu"
    return False


def is_device_cuda(device: Union[None, str, torch.device] = None) -> bool:
    if device is None:
        return False
    if not torch.cuda.is_available():
        return False
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return "cuda" in device
    return False


def is_device_mps(device: Union[None, str, torch.device] = None) -> bool:
    if device is None:
        return False
    if not torch.backends.mps.is_available():
        return False
    if isinstance(device, torch.device):
        return device.type == "mps"
    if isinstance(device, str):
        return "mps" in device
    return False


def get_device_memory(device: Union[None, str, torch.device] = None) -> float:
    """ Returns the amount of memory available for this device in GB. If it was not possible to calculate, 4 GB is
     used as a conservative default and 8 GB for mps """
    gb_factor = (1024 ** 3)
    conservative_default = 4

    if device is None or isinstance(device, str):
        device = get_device()
    if is_device_cuda(device):
        free_mem, _ = torch.cuda.mem_get_info(device=device)
        return free_mem / gb_factor
    if is_device_cpu(device):
        vm = psutil.virtual_memory()
        return vm.available / gb_factor
    if is_device_mps(device):
        # Get recommended max working set size from Metal
        recommended_max = torch.mps.recommended_max_memory()

        # Get currently allocated memory by PyTorch tensors
        current_allocated = torch.mps.current_allocated_memory()

        # Calculate available memory
        available_bytes = recommended_max - current_allocated
        available_gb = available_bytes / gb_factor

        # Ensure we return at least the conservative default
        return max(available_gb, conservative_default)

    return conservative_default
