from .config_option import ConfigOption, ConfigConstraints

from ..protocols import Protocol
from ..embedders import get_predefined_embedder_names

def embedding_config(protocol: Protocol):
    return "", [
        ConfigOption(
            name="embedder_name",
            type=str,
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
            required=False,
            default=False
        ),
        ConfigOption(
            name="embeddings_file",
            type=str,
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
            required=False,
            default=None,
            constraints=ConfigConstraints(allowed_protocols=Protocol.using_per_sequence_embeddings(),
                                          gt=0)
        )
    ]
