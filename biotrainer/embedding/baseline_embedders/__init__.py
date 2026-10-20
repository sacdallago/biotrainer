from typing import Callable, Dict, Optional

from ..interfaces import BaselineEmbedder
from .random_embedder import RandomEmbedder
from .constant_embedder import ConstantEmbedder
from .length_embedder import LengthEmbedder
from .blosum62_embedder import Blosum62Embedder
from .aa_ontology_embedder import AAOntologyEmbedder
from .one_hot_encoding_embedder import OneHotEncodingEmbedder

# Each entry builds its embedder from an optional seed, which only embedders that draw at random use
BASELINE_EMBEDDERS: Dict[str, Callable[[Optional[int]], BaselineEmbedder]] = {
    "one_hot_encoding": lambda _: OneHotEncodingEmbedder(),
    "random_embedder": lambda seed: RandomEmbedder(seed=seed),
    "constant_embedder": lambda _: ConstantEmbedder(),
    "AAOntology": lambda _: AAOntologyEmbedder(),
    "blosum62": lambda _: Blosum62Embedder(),
    "length_embedder": lambda _: LengthEmbedder(),
}

__all__ = ["BASELINE_EMBEDDERS", "AAOntologyEmbedder", "OneHotEncodingEmbedder", "RandomEmbedder", "ConstantEmbedder",
           "Blosum62Embedder", "LengthEmbedder"]
