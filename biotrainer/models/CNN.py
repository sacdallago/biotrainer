import torch.nn as nn


# Convolutional neural network (two convolutional layers)
class CNN(nn.Module):
    def __init__(
            self, n_classes: int, n_features: int, pretrained_model=None,
            bottleneck_dim: int = 32
    ):
        super(CNN, self).__init__()

        # dropout_rate = 0.25
        self.classifier = nn.Sequential(
            nn.Conv2d(n_features, bottleneck_dim, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Conv2d(bottleneck_dim, n_classes, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (9 for conservation)
        """
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)  # IN: X = (B x L x F); OUT: (B x F x L, 1)
        Yhat = self.classifier(x)  # OUT: Yhat_consurf = (B x N x L x 1)
        Yhat = Yhat.squeeze(dim=-1)  # IN: (B x N x L x 1); OUT: ( B x L x N )
        return Yhat