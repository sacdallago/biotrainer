import torch
from torch.nn import Parameter
from typing import Iterator, Tuple

from .biotrainer_model import BiotrainerModel

from ..protocols import Protocol
from ..embedders import PeftEmbeddingService


class FineTuningModel(BiotrainerModel):
    def __init__(self, embedding_service: PeftEmbeddingService, downstream_model, collate_fn, protocol, device):
        super().__init__()
        self.embedding_service = embedding_service
        self.downstream_model = downstream_model

        self.collate_fn = collate_fn
        self.reduced_embeddings = protocol in Protocol.using_per_sequence_embeddings()
        self.device = device

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        assert self.embedding_service._embedder._model, f"Trying to get parameters of non-existing embedder model!"
        params = list(self.embedding_service._embedder._model.parameters()) + list(self.downstream_model.parameters())
        return params

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        assert self.embedding_service._embedder._model, f"Trying to get parameters of non-existing embedder model!"
        params = list(self.embedding_service._embedder._model.named_parameters(prefix, recurse, remove_duplicate)) + list(self.downstream_model.named_parameters(prefix, recurse, remove_duplicate))
        return params

    def get_downstream_model(self):
        return self.downstream_model

    def eval(self):
        self.embedding_service._embedder._model = self.embedding_service._embedder._model.eval()
        self.downstream_model = self.downstream_model.eval()
        return self

    def train(self, mode: bool = True):
        self.embedding_service._embedder._model = self.embedding_service._embedder._model.train()
        self.downstream_model = self.downstream_model.train()
        return self

    def forward(self, sequences, *args, **kwargs) -> Tuple:
        # Compute embeddings with gradients
        targets = kwargs.pop('targets', [])

        embeddings = self.embedding_service.generate_embeddings(input_data=sequences,
                                                                reduce=self.reduced_embeddings)

        downstream_batch = [(seq_record.seq_id, embedding, targets[idx]) for idx, (seq_record, embedding)
                            in
                            enumerate(embeddings)]
        _, downstream_embeddings, targets, lengths = self.collate_fn(downstream_batch)
        downstream_input = torch.stack([embedding.to(self.device) for embedding in downstream_embeddings])
        padded_targets = torch.stack([target.to(self.device) for target in targets])
        return self.downstream_model(downstream_input, *args, **kwargs), padded_targets
