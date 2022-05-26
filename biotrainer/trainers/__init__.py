from .Trainer import Trainer
from .EmbeddingsLoader import EmbeddingsLoader
from .PredictionIOHandler import PredictionIOHandler


def get_trainer(**kwargs):
    embeddings_loader = EmbeddingsLoader(**kwargs)
    prediction_io_handler = PredictionIOHandler(**kwargs)
    trainer = Trainer(embeddings_loader, prediction_io_handler, **kwargs)

    if not trainer:
        raise NotImplementedError
    else:
        return trainer.pipeline()


__all__ = [
    'get_trainer'
]
