from typing import List, Any, Union, Type
from abc import ABC

from .config_option import ConfigOption, ConfigurationException, classproperty
from ..protocols import Protocol


class HFDatasetOption(ConfigOption, ABC):
    """
    Abstract base class for HuggingFace dataset configuration options.

    Extends `ConfigOption` to provide a framework for defining
    specific HuggingFace dataset-related options.
    """

    @classproperty
    def category(self) -> str:
        return "hf_dataset"


class HFPath(HFDatasetOption):
    """
    Configuration option for specifying the HuggingFace dataset path.
    """

    @classproperty
    def name(self) -> str:
        return "path"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @classproperty
    def required(self) -> bool:
        return True

    def check_value(self) -> bool:
        if not isinstance(self.value, str) or "/" not in self.value:
            raise ConfigurationException(
                f"Invalid HuggingFace dataset path: {self.value}. It should be in the format 'username/dataset_name'."
            )
        return True


class HFSubsetName(HFDatasetOption):
    """
    Configuration option for specifying the dataset subset name.
    """

    @classproperty
    def name(self) -> str:
        return "subset"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @classproperty
    def required(self) -> bool:
        return False


class HFSequenceColumn(HFDatasetOption):
    """
    Configuration option for specifying the sequence column in the dataset.
    """

    @classproperty
    def name(self) -> str:
        return "sequence_column"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @classproperty
    def required(self) -> bool:
        return True

    def check_value(self) -> bool:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ConfigurationException("sequence_column must be a non-empty string.")
        return True


class HFTargetColumn(HFDatasetOption):
    """
    Configuration option for specifying the target column in the dataset.
    """

    @classproperty
    def name(self) -> str:
        return "target_column"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @classproperty
    def required(self) -> bool:
        return True

    def check_value(self) -> bool:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ConfigurationException("target_column must be a non-empty string.")
        return True

class HFMaskColumn(HFDatasetOption):
    """
    Configuration option for specifying the mask column in the dataset.
    """

    @classproperty
    def name(self) -> str:
        return "mask_column"

    @classproperty
    def allow_multiple_values(self) -> bool:
        return False

    @classproperty
    def required(self) -> bool:
        return False

    def check_value(self) -> bool:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ConfigurationException("mask_column must be a non-empty string.")
        return True

# Constant key for hf_dataset configuration
HF_DATASET_CONFIG_KEY: str = "hf_dataset"

# Add hf_dataset options to a separate dictionary
hf_dataset_options: List[Type[HFDatasetOption]] = [
    HFDatasetOption,
    HFPath,
    HFSubsetName,
    HFSequenceColumn,
    HFTargetColumn,
    HFMaskColumn
]
