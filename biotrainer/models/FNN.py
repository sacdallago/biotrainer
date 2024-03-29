import torch.nn as nn


# Feed-Forward Neural Network (FNN) with two linear layers connected by a non-lin
class FNN(nn.Module):
    def __init__(
            self, n_classes: int, n_features: int,
            bottleneck_dim: int = 32, dropout_rate: float = 0.25,
            **kwargs
    ):
        super(FNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(n_features, bottleneck_dim),  # n_features x 32
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, n_classes)
        )

    def forward(self, x):
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (9 for conservation)
        """
        # IN: X = (B x L x F)
        Yhat = self.classifier(x)  # OUT: Yhat_consurf = (B x L x N)
        return Yhat


class DeeperFNN(FNN):
    def __init__(
            self, n_classes: int, n_features: int,
            bottleneck_dim: int = 128, dropout_rate: float = 0.25,
            **kwargs
    ):
        super(FNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(n_features, bottleneck_dim),  # n_features x 128 (for default)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, int(bottleneck_dim / 2)),  # 128 x 64 (for default)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(bottleneck_dim / 2), int(bottleneck_dim / 4)),  # 64 x 32 (for default)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(bottleneck_dim / 4), n_classes)  # 32 x n_classes (for default)
        )
