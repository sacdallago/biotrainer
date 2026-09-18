from typing import List

from .onnx import OnnxEmbedder
from .embedding_api import EmbeddingAPI
from .stats import EmbeddingStatsTracker, EmbeddingStats
from .interfaces import EmbedderInterface, CustomTokenizer, CustomEmbedder
from .services import EmbeddingService, PeftEmbeddingService, get_embedding_service
from .huggingface import HuggingfaceTransformerEmbedder, ProtT5, ProstT5, ESM2
from .baseline_embedders import RandomEmbedder, AAOntologyEmbedder, OneHotEncodingEmbedder, Blosum62Embedder, BASELINE_EMBEDDERS


def get_predefined_embedder_names() -> List[str]:
    return list(BASELINE_EMBEDDERS.keys())


__all__ = [
    "EmbeddingAPI",
    "EmbedderInterface",
    "EmbeddingService",
    "PeftEmbeddingService",
    "EmbeddingStatsTracker",
    "EmbeddingStats",
    "OneHotEncodingEmbedder",
    "RandomEmbedder",
    "AAOntologyEmbedder",
    "get_embedding_service",
    "get_predefined_embedder_names",
    "CustomEmbedder",
]
