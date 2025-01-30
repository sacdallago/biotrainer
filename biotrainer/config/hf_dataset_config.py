from typing import Tuple, List

from .config_option import ConfigOption, ConfigConstraints, ConfigKey

from ..protocols import Protocol


def hf_dataset_config(protocol: Protocol) -> Tuple[ConfigKey, List[ConfigOption]]:
    hf_dataset_category = "hf_dataset"
    return ConfigKey.HF_DATASET, [
        ConfigOption(
            name="path",
            description="Specify the path to the HuggingFace dataset.",
            category=hf_dataset_category,
            required=True,
            constraints=ConfigConstraints(
                type=str,
                # Custom validation for HuggingFace dataset path format
                custom_validator=lambda value: ("/" in value, "Missing / in value!")
            )
        ),
        ConfigOption(
            name="subset",
            description="Specify the subset of the employed HuggingFace dataset.",
            category=hf_dataset_category,
            required=False,
            constraints=ConfigConstraints(
                type=str,
            )
        ),
        ConfigOption(
            name="sequence_column",
            description="Specify the sequence column name in the HuggingFace dataset.",
            category=hf_dataset_category,
            required=True,
            constraints=ConfigConstraints(
                type=str,
                # Ensure non-empty string
                custom_validator=lambda value: (bool(value and value.strip()), "No value given!")
            )
        ),
        ConfigOption(
            name="target_column",
            description="Specify the target column name in the HuggingFace dataset.",
            category=hf_dataset_category,
            required=True,
            constraints=ConfigConstraints(
                type=str,
                # Ensure non-empty string
                custom_validator=lambda value: (bool(value and value.strip()), "No value given!")
            )
        ),
        ConfigOption(
            name="mask_column",
            description="Specify the mask column name in the HuggingFace dataset.",
            category=hf_dataset_category,
            required=False,
            constraints=ConfigConstraints(
                type=str,
                # Ensure non-empty string if provided
                custom_validator=lambda value: value is None or bool(value and value.strip())
            )
        )
    ]