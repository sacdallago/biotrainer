import torch

from torch import nn
from torch.nn import Parameter
from typing import Iterator, Union, Tuple


class BiotrainerModel(nn.Module):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> Union[torch.Tensor, Tuple]:
        """Returns downstream model predictions (and padded targets for finetuning model)"""
        raise NotImplementedError

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return super().parameters(recurse)

    def get_downstream_model(self):
        return self

    def eval(self):
        return super().eval()

    def train(self, mode: bool = True):
        return super().train(mode)

    def compile(self):
        # Using TensorFloat32 tensor cores is suggested when using a compiled model:
        torch.set_float32_matmul_precision('high')
        super().compile(backend="aot_eager")