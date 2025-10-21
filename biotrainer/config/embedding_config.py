from typing import List, Tuple

from .config_option import ConfigOption, ConfigConstraints, ConfigKey

from ..protocols import Protocol
from ..embedders import get_predefined_embedder_names


def embedding_config(protocol: Protocol) -> Tuple[ConfigKey, List[ConfigOption]]:
    embedding_category = "embedding"
    return ConfigKey.ROOT, [
        ConfigOption(
            name="embedder_name",
            description=f"Allows to define which embedder to use. Can be either a predefined embedder "
                        f"{get_predefined_embedder_names()}, a .onnx model or a huggingface transformer model name.",
            category=embedding_category,
            required=False,
            default="custom_embeddings",
            constraints=ConfigConstraints(
                type=str,
            ),
        ),
        ConfigOption(
            name="custom_tokenizer_config",
            category=embedding_category,
            description="Provide a json config file containing information for a custom tokenizer",
            required=False,
            is_file_option=True,
            default=None,
            constraints=ConfigConstraints(
                type=str,
                allowed_formats=[".json"]
            )
        ),
        ConfigOption(
            name="use_half_precision",
            description="Enable or disable half-precision embedding computations, which can lead to "
                        "performance improvements and reduced memory usage at the potential "
                        "cost of numerical precision. Only applicable for embedding on the GPU.",
            category=embedding_category,
            required=False,
            default=False,
            constraints=ConfigConstraints(
                type=bool,
            )
        ),
        ConfigOption(
            name="embeddings_file",
            category=embedding_category,
            description="Provide a file containing precomputed embeddings, typically in HDF5 (`.h5`) format.",
            required=False,
            is_file_option=True,
            default=None,
            constraints=ConfigConstraints(
                type=str,
                allowed_formats=[".h5", ".hdf5"]
            )
        ),
        ConfigOption(
            name="dimension_reduction_method",
            description="Choose an optional method to reduce embedding dimensions.",
            category=embedding_category,
            required=False,
            default=None,
            constraints=ConfigConstraints(
                type=str,
                allowed_protocols=Protocol.using_per_sequence_embeddings(),
                allowed_values=["umap", "pca"] if protocol in Protocol.using_per_sequence_embeddings() else []
            )
        ),
        ConfigOption(
            name="n_reduced_components",
            description="Choose the number of reduced components to use for the dimension reduction method.",
            category=embedding_category,
            required=False,
            default=None,
            constraints=ConfigConstraints(
                type=int,
                allowed_protocols=Protocol.using_per_sequence_embeddings(),
                gt=0
            )
        )
    ]
