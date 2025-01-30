from typing import Tuple, List

from .config_option import ConfigOption, ConfigConstraints, ConfigKey

from ..protocols import Protocol

def input_config(protocol: Protocol) -> Tuple[ConfigKey, List[ConfigOption]]:
    input_category = "input_files"
    return ConfigKey.ROOT, [
        ConfigOption(
            name="sequence_file",
            description="Provide the sequence file in FASTA format, "
                        "which contains the biological sequences to be used for training.",
            category=input_category,
            required=False,
            is_file_option=True,
            constraints=ConfigConstraints(
                type=str,
                allowed_formats=[".fasta"]
            )
        ),
        ConfigOption(
            name="labels_file",
            description="Provide a labels file in FASTA format, "
                        "which contains the labels corresponding to residues for each sequence.",
            category=input_category,
            required=False,
            is_file_option=True,
            constraints=ConfigConstraints(
                type=str,
                allowed_formats=[".fasta"],
                allowed_protocols=Protocol.per_residue_protocols()
            )
        ),
        ConfigOption(
            name="mask_file",
            description="Provide the mask file in FASTA format, "
                        "which contains the mask corresponding to residues for each sequence.",
            category=input_category,
            required=False,
            is_file_option=True,
            constraints=ConfigConstraints(
                type=str,
                allowed_formats=[".fasta"],
                allowed_protocols=Protocol.per_residue_protocols()
            )
        )
    ]