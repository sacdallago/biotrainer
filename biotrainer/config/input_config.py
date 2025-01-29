from .config_option import ConfigOption, ConfigConstraints

from ..protocols import Protocol

def input_config(protocol: Protocol):
    input_category = "input_files"
    return "", [
        ConfigOption(
            name="sequence_file",
            type=str,
            description="Provide the sequence file in FASTA format, "
                        "which contains the biological sequences to be used for training.",
            category=input_category,
            required=False,
            is_file_option=True,
            constraints=ConfigConstraints(
                allowed_formats=[".fasta"]
            )
        ),
        ConfigOption(
            name="labels_file",
            type=str,
            description="Provide a labels file in FASTA format, "
                        "which contains the labels corresponding to residues for each sequence.",
            category=input_category,
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
            description="Provide the mask file in FASTA format, "
                        "which contains the mask corresponding to residues for each sequence.",
            category=input_category,
            required=False,
            is_file_option=True,
            constraints=ConfigConstraints(
                allowed_formats=[".fasta"],
                allowed_protocols=Protocol.per_residue_protocols()
            )
        )
    ]