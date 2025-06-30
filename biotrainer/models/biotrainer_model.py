from typing import Iterator

import torch

from torch import nn
from torch.nn import Parameter


class BiotrainerModel(nn.Module):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return super().parameters(recurse)

    def get_downstream_model(self):
        return self

    def eval(self):
        return super().eval()

    def train(self, mode: bool = True):
        return super().train(mode)