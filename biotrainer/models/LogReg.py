import torch.nn as nn


# Logistic regression (single linear layer directly mapping to classes)
class LogReg(nn.Module):
    def __init__(self, n_classes: int, n_features: int, pretrained_model=None):
        super(LogReg, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_features, n_classes), # 7x32
        )

    def forward( self, x):
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (9 for conservation)
        """
        Yhat = self.classifier(x)  # IN: X = (B x L x F); OUT: Yhat_consurf = (B x L x N)
        return Yhat


class LogRegInteraction(nn.Module):
    def __init__(self, n_classes: int, n_features: int, pretrained_model=None):
        super(LogRegInteraction, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 1),  # 7x32
        )

    def forward(self, x):
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (9 for conservation)
        """
        Yhat = self.classifier(x)  # IN: X = (B x L x F); OUT: Yhat_consurf = (B x L x N)
        return Yhat