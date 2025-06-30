import torch
import torch.nn as nn

import biotrainer.utilities as utils

from .biotrainer_model import BiotrainerModel


# LightAttention model originally from Hannes Stark:
# https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
class LightAttention(BiotrainerModel):
    def __init__(self, n_features: int, n_classes: int, dropout_rate=0.25, kernel_size=9, conv_dropout: float = 0.25,
                 **kwargs
                 ):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(n_features, n_features, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(n_features, n_features, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * n_features, 32),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        self.output = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """

        # Calculate mask (== where all residues are utils.SEQUENCE_PAD_VALUE)
        mask = x.sum(dim=-1) != utils.SEQUENCE_PAD_VALUE

        x = x.permute(0, 2, 1)
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]
        o = o * mask.unsqueeze(1)

        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention
        # (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the pad_residues_embeddings collate function
        attention = attention.masked_fill(mask[:, None, :] == False, -float('inf'))

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]
