from .config_option import ConfigOption, ConfigConstraints

from ..protocols import Protocol

def input_config(protocol: Protocol):
    return "", [
        ConfigOption(
            name="sequence_file",
            type=str,
            required=False,
            is_file_option=True,
            constraints=ConfigConstraints(
                allowed_formats=[".fasta"]
            )
        ),
        ConfigOption(
            name="labels_file",
            type=str,
            required=False,
            is_file_option=True,
            constraints=ConfigConstraints(
                allowed_formats=[".fasta"],
                allowed_protocols=Protocol.per_residue_protocols()
            )
        ),
        ConfigOption(
            name="mask_file",
            type=str,
            required=False,
            is_file_option=True,
            constraints=ConfigConstraints(
                allowed_formats=[".fasta"],
                allowed_protocols=Protocol.per_residue_protocols()
            )
        )
    ]