from typing import Tuple, List

from .config_option import ConfigOption, ConfigConstraints, ConfigKey

from ..protocols import Protocol

def input_config(protocol: Protocol) -> Tuple[ConfigKey, List[ConfigOption]]:
    input_category = "input_files"
    return ConfigKey.ROOT, [
        ConfigOption(
            name="input_file",
            description="Provide the input file, which contains the biological sequences, "
                        "targets and optionally masks to be used for training.",
            category=input_category,
            required=False,  # Can be replaced by hf_dataset config
            is_file_option=True,
            constraints=ConfigConstraints(
                type=str,
                allowed_formats=[".fasta"]
            )
        ),
        ConfigOption(
            name="input_data",
            description="Provide the input data directly as BiotrainerSequenceRecords.",
            category=input_category,
            required=False,  # Can be replaced by hf_dataset config
            is_file_option=False,
            constraints=ConfigConstraints(
                type=List,
            )
        ),
    ]