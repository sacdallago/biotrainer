import torch
import torch.nn as nn

from .biotrainer_model import BiotrainerModel


# Logistic regression (single linear layer directly mapping to classes)
class LogReg(BiotrainerModel):
    def __init__(self, n_classes: int, n_features: int, **kwargs):
        super(LogReg, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_features, n_classes),  # 7x32
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (9 for conservation)
        """
        Yhat = self.classifier(x)  # IN: X = (B x L x F); OUT: Yhat_consurf = (B x L x N)
        return Yhat
