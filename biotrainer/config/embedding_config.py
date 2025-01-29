from .config_option import ConfigOption, ConfigConstraints

from ..protocols import Protocol
from ..embedders import get_predefined_embedder_names


def embedding_config(protocol: Protocol):
    embedding_category = "embedding"
    return "", [
        ConfigOption(
            name="embedder_name",
            type=str,
            description=f"Allows to define which embedder to use. Can be either a predefined embedder "
                        f"{get_predefined_embedder_names()} or a huggingface transformer model name.",
            category=embedding_category,
            required=False,
            default="custom_embeddings",
            constraints=ConfigConstraints(
                # Validate embedder name or file path
                custom_validator=lambda value: (
                    value in get_predefined_embedder_names() or
                    value == "custom_embeddings" or
                    value.endswith(".py") or
                    "/" in value, "Could not find given embedder!")
            ),
        ),
        ConfigOption(
            name="use_half_precision",
            type=bool,
            description="Enable or disable half-precision embedding computations, which can lead to "
                        "performance improvements and reduced memory usage at the potential "
                        "cost of numerical precision. Only applicable for embedding on the GPU.",
            category=embedding_category,
            required=False,
            default=False
        ),
        ConfigOption(
            name="embeddings_file",
            type=str,
            category=embedding_category,
            description="Provide a file containing precomputed embeddings, typically in HDF5 (`.h5`) format.",
            required=False,
            is_file_option=True,
            default=None,
            constraints=ConfigConstraints(
                allowed_formats=[".h5", ".hdf5"]
            )
        ),
        ConfigOption(
            name="dimension_reduction_method",
            type=str,
            description="Choose an optional method to reduce embedding dimensions.",
            category=embedding_category,
            required=False,
            default=None,
            constraints=ConfigConstraints(
                allowed_protocols=Protocol.using_per_sequence_embeddings(),
                allowed_values=["umap", "tsne"] if protocol in Protocol.using_per_sequence_embeddings() else []
            )
        ),
        ConfigOption(
            name="n_reduced_components",
            type=int,
            description="Choose the number of reduced components to use for the dimension reduction method.",
            category=embedding_category,
            required=False,
            default=None,
            constraints=ConfigConstraints(allowed_protocols=Protocol.using_per_sequence_embeddings(),
                                          gt=0)
        )
    ]
