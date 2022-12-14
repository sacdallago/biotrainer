import torch
import torch.nn as nn

from typing import Optional
from ..losses import get_loss
from ..models import get_model
from ..optimizers import get_optimizer


class ModelFactory:
    __slots__ = ["protocol", "model_choice", "loss_choice", "optimizer_choice", "learning_rate", "device"]

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.__slots__:
                setattr(self, key, value)

    def create_model_loss_optimizer(self, n_classes: int, n_features: int,
                                    class_weights: Optional[torch.Tensor] = None) -> \
            (nn.Module, nn.Module, nn.Module):
        # Initialize model
        model = get_model(
            protocol=self.protocol, model_choice=self.model_choice,
            n_classes=n_classes, n_features=n_features
        )

        # Initialize loss function
        loss_function = get_loss(
            protocol=self.protocol, loss_choice=self.loss_choice, device=self.device, weight=class_weights
        )

        # Initialize optimizer
        optimizer = get_optimizer(
            protocol=self.protocol, optimizer_choice=self.optimizer_choice,
            learning_rate=self.learning_rate, model_parameters=model.parameters()
        )

        return model, loss_function, optimizer
