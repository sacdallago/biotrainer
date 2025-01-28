from .config_option import ConfigOption, ConfigConstraints

from ..protocols import Protocol


def hf_dataset_config(protocol: Protocol):
    return "hf_dataset", [
        ConfigOption(
            name="path",
            type=str,
            required=True,
            constraints=ConfigConstraints(
                # Custom validation for HuggingFace dataset path format
                custom_validator=lambda value: ("/" in value, "Missing / in value!")
            )
        ),
        ConfigOption(
            name="subset",
            type=str,
            required=False
        ),
        ConfigOption(
            name="sequence_column",
            type=str,
            required=True,
            constraints=ConfigConstraints(
                # Ensure non-empty string
                custom_validator=lambda value: (bool(value and value.strip()), "No value given!")
            )
        ),
        ConfigOption(
            name="target_column",
            type=str,
            required=True,
            constraints=ConfigConstraints(
                # Ensure non-empty string
                custom_validator=lambda value: (bool(value and value.strip()), "No value given!")
            )
        ),
        ConfigOption(
            name="mask_column",
            type=str,
            required=False,
            constraints=ConfigConstraints(
                # Ensure non-empty string if provided
                custom_validator=lambda value: value is None or bool(value and value.strip())
            )
        )
    ]