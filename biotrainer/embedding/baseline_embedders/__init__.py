from .random_embedder import RandomEmbedder
from .constant_embedder import ConstantEmbedder
from .length_embedder import LengthEmbedder
from .blosum62_embedder import Blosum62Embedder
from .aa_ontology_embedder import AAOntologyEmbedder
from .one_hot_encoding_embedder import OneHotEncodingEmbedder

BASELINE_EMBEDDERS = {
    "one_hot_encoding": OneHotEncodingEmbedder,
    "random_embedder": RandomEmbedder,
    "constant_embedder": ConstantEmbedder,
    "AAOntology": AAOntologyEmbedder,
    "blosum62": Blosum62Embedder,
    "length_embedder": LengthEmbedder,
}

__all__ = ["BASELINE_EMBEDDERS", "AAOntologyEmbedder", "OneHotEncodingEmbedder", "RandomEmbedder", "ConstantEmbedder",
           "Blosum62Embedder", "LengthEmbedder"]
