import torch.nn as nn


def count_parameters(model: nn.Module):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
