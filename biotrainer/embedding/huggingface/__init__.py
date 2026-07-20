from .ESM2 import ESM2, ESMC
from .ProtT5 import ProtT5
from .ProstT5 import ProstT5
from .huggingface_transformer_embedder import HuggingfaceTransformerEmbedder

OPTIMIZED_EMBEDDERS = [
    ProtT5,
    ProstT5,
    ESM2,
    ESMC,
]

__all__ = ["OPTIMIZED_EMBEDDERS", "HuggingfaceTransformerEmbedder", "ProtT5", "ProstT5", "ESM2", "ESMC"]
