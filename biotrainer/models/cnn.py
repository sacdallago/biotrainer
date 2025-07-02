import torch
import torch.nn as nn
import biotrainer.utilities as utils

from .biotrainer_model import BiotrainerModel


# Convolutional neural network (two convolutional layers)
class CNN(BiotrainerModel):
    def __init__(
            self, n_classes: int, n_features: int,
            bottleneck_dim: int = 32,
            dropout_rate: float = 0.0,  # Dropout for CNN disabled by default
            **kwargs,
    ):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(n_features, bottleneck_dim, kernel_size=(7, 1), padding=(3, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(bottleneck_dim, n_classes, kernel_size=(7, 1), padding=(3, 0))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (9 for conservation)
        """

        # Calculate mask
        mask = (x.sum(dim=-1) != utils.SEQUENCE_PAD_VALUE).unsqueeze(1).unsqueeze(3)  # Shape: (B, 1, L, 1)

        x = x.permute(0, 2, 1).unsqueeze(3)  # Shape: (B, F, L, 1)

        # First convolution
        x = self.conv1(x)
        x = self.relu(x)
        x = x * mask  # Apply mask

        # Dropout
        x = self.dropout(x)

        # Second convolution
        x = self.conv2(x)
        x = x * mask  # Apply mask

        # Remove the last dimension and permute back
        x = x.squeeze(3).permute(0, 1, 2)  # Shape: (B, L, N)

        return x
