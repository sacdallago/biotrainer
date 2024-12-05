import os
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split

from ruamel import yaml
from ruamel.yaml import YAMLError
from webencodings import labels

from . import config_rules
from .config_option import ConfigurationException, ConfigOption, FileOption, logger
from .config_rules import (
    ConfigRule,
    MutualExclusive,
    MutualExclusiveValues,
    ProtocolRequires,
    OptionValueRequires,
    AllowHyperparameterOptimization,
)
from .cross_validation_options import (
    cross_validation_options,
    CROSS_VALIDATION_CONFIG_KEY,
    Method,
    ChooseBy,
    K,
    Nested,
    NestedK,
    SearchMethod,
    NMaxEvaluationsRandom,
    P,
)
from .embedding_options import EmbedderName, EmbeddingsFile, embedding_options, UseHalfPrecision
from .general_options import general_options, Device
from .input_options import SequenceFile, LabelsFile, MaskFile, input_options
from .model_options import model_options
from .training_options import AutoResume, PretrainedModel, training_options
from .hf_dataset_options import (
    hf_dataset_options,
    HF_DATASET_CONFIG_KEY,
    HFPath,
    HFSequenceColumn,
    HFTargetColumn,
)
from ..protocols import Protocol
from ..utilities import process_hf_dataset_to_fasta

# Define protocol-specific rules
local_file_protocol_rules = [
    ProtocolRequires(protocol=Protocol.per_residue_protocols(), requires=[SequenceFile, LabelsFile]),
    ProtocolRequires(protocol=Protocol.per_sequence_protocols(), requires=[SequenceFile]),
]

hf_dataset_rules = [
    ProtocolRequires(protocol=Protocol.all(), requires=[HFPath, HFSequenceColumn, HFTargetColumn]),
    MutualExclusive(
        exclusive=[HFPath, SequenceFile],
        error_message="If you want to download from HuggingFace, don't provide a sequence_file.\n"
                      "Providing sequence_column is enough to download the dataset from HuggingFace."
    ),
    MutualExclusive(
            exclusive=[HFPath, LabelsFile],
            error_message="If you want to download from HuggingFace, don't provide a labels_file.\n"
                          "Providing targets_column is enough to download the dataset from HuggingFace."
        ),
    MutualExclusive(
            exclusive=[HFPath, MaskFile],
            error_message="If you want to download from HuggingFace, don't provide a mask_file.\n"
                          "Providing mask_column is enough to download the dataset from HuggingFace."
        )
]

# Define configuration option rules
config_option_rules = [
    MutualExclusive(
        exclusive=[AutoResume, PretrainedModel],
        error_message=(
            "Use auto_resume in case you need to restart your training job multiple times.\n"
            "Use pretrained_model if you want to continue to train a specific model."
        ),
    ),
    MutualExclusive(
        exclusive=[EmbedderName, EmbeddingsFile],
        allowed_values=["custom_embeddings"],
        error_message=(
            "Please provide either an embedder_name to calculate embeddings from scratch or \n"
            "an embeddings_file to use pre-computed embeddings."
        ),
    ),
    MutualExclusiveValues(
        exclusive={UseHalfPrecision: True, Device: "cpu"},
        error_message=(
            "use_half_precision mode is not compatible with embedding on the CPU. "
            "(See: https://github.com/huggingface/transformers/issues/11546)"
        ),
    ),
]

# Define optimization rules
optimization_rules = [
    AllowHyperparameterOptimization(option=Nested, value=True),
]

# Define cross-validation rules
cross_validation_rules = [
    OptionValueRequires(option=Method, value="k_fold", requires=[K]),
    OptionValueRequires(option=Nested, value=True, requires=[NestedK, SearchMethod]),
    OptionValueRequires(option=SearchMethod, value="random_search", requires=[NMaxEvaluationsRandom]),
    OptionValueRequires(option=Method, value="leave_p_out", requires=[P]),
]

# Combine all configuration options into dictionaries for easy access
all_options_dict: Dict[str, ConfigOption] = {
    option.name: option
    for option in (
        general_options + input_options + hf_dataset_options +
        model_options + training_options + embedding_options
    )
}

cross_validation_dict: Dict[str, ConfigOption] = {
    option.name: option for option in cross_validation_options
}

hf_dataset_dict: Dict[str, ConfigOption] = {
    option.name: option for option in hf_dataset_options
}


class Configurator:
    """
    Class to read, validate, and transform the input YAML configuration.

    The `Configurator` class handles the parsing of configuration files, applies validation rules,
    transforms configuration values (such as downloading necessary files), and ensures that the
    configuration is consistent and adheres to the defined rules based on the selected protocol.
    """

    def __init__(
        self,
        config_dict: Dict,
        config_file_path: Path = None
    ):
        """
        Initialize a Configurator instance.

        Args:
            config_dict (Dict): The configuration dictionary parsed from the YAML file.
            config_file_path (Path, optional): Path to the configuration file. Defaults to the current directory.
        """
        if not config_file_path:
            config_file_path = Path("")
        self._config_file_path = config_file_path
        self._config_dict = config_dict
        self.protocol = self._get_protocol_from_config_dict(config_dict)

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]):
        """
        Create a Configurator instance from a configuration dictionary.

        Args:
            config_dict (Dict[str, Any]): The configuration dictionary.

        Returns:
            Configurator: An instance of the Configurator class initialized with the provided dictionary.
        """
        return cls(config_dict=config_dict)

    @classmethod
    def from_config_path(cls, config_path: Union[str, Path]):
        """
        Create a Configurator instance by reading a configuration file.

        Args:
            config_path (Union[str, Path]): Path to the YAML configuration file.

        Returns:
            Configurator: An instance of the Configurator class initialized with the configuration file.
        """
        return cls(
            config_dict=cls._read_config_file(config_path),
            config_file_path=Path(os.path.dirname(os.path.abspath(config_path))),
        )

    @staticmethod
    def get_option_dicts_by_protocol(
        protocol: Protocol,
        include_cross_validation_options: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Returns all possible configuration options as dictionaries for the given protocol.

        Args:
            protocol (Protocol): The protocol to get all options for.
            include_cross_validation_options (bool, optional): If True, includes cross-validation options. Defaults to False.

        Returns:
            List[Dict[str, Any]]: A list of all configuration options as dictionaries.
        """
        result = []
        all_config_options_dict = (
            all_options_dict | cross_validation_dict
            if include_cross_validation_options
            else all_options_dict
        )
        for option_class in all_config_options_dict.values():
            option = option_class(protocol=protocol)
            if protocol in option.allowed_protocols:
                result.append(option.to_dict())
        return result

    @staticmethod
    def _read_config_file(
        config_path: Union[str, Path],
        preserve_order: bool = True
    ) -> dict:
        """
        Read configuration from a YAML file.

        Args:
            config_path (Union[str, Path]): Path to the YAML configuration file.
            preserve_order (bool, optional): Whether to preserve the order of the YAML file. Defaults to True.

        Returns:
            dict: The configuration file parsed as a dictionary.

        Raises:
            ConfigurationException: If the YAML file cannot be parsed.
        """
        with open(config_path, "r") as fp:
            try:
                if preserve_order:
                    return yaml.load(fp, Loader=yaml.Loader)
                else:
                    return yaml.safe_load(fp)
            except YAMLError as e:
                raise ConfigurationException(
                    f"Could not parse configuration file at '{config_path}' as YAML. "
                    "Formatting mistake in config file? "
                    "See error above for details."
                ) from e

    @staticmethod
    def _get_protocol_from_config_dict(config_dict: Dict[str, Any]):
        """
        Extract the protocol from the configuration dictionary.

        Args:
            config_dict (Dict[str, Any]): The configuration dictionary.

        Returns:
            Protocol: The extracted protocol.

        Raises:
            ConfigurationException: If the protocol is not specified or invalid.
        """
        protocol = config_dict.get("protocol")
        if protocol is None:
            raise ConfigurationException(
                "No protocol specified in config file!"
            )
        try:
            return Protocol[protocol]
        except KeyError:
            raise ConfigurationException(
                f"Invalid protocol specified: {protocol}"
            )

    @staticmethod
    def _get_cross_validation_map(
        protocol: Protocol,
        cv_dict: Dict[str, Any]
    ) -> Dict[str, ConfigOption]:
        """
        Create a mapping of cross-validation options based on the protocol and provided configuration.

        Args:
            protocol (Protocol): The selected protocol.
            cv_dict (Dict[str, Any]): The cross-validation configuration dictionary.

        Returns:
            Dict[str, ConfigOption]: A dictionary mapping cross-validation option names to their instances.

        Raises:
            ConfigurationException: If an unknown cross-validation option is encountered or required options are missing.
        """
        cv_map = {}
        method = ""
        for cv_name in cv_dict.keys():
            try:
                value = cv_dict[cv_name]
                if cv_name == Method.name:
                    method = value
                cv_object: ConfigOption = cross_validation_dict[cv_name](protocol=protocol, value=value)
                cv_object.transform_value_if_necessary()
                cv_map[cv_name] = cv_object
            except KeyError:
                raise ConfigurationException(
                    f"Unknown cross-validation option: {cv_name}!"
                )

        if method == "":
            raise ConfigurationException(
                "Required option method is missing from cross_validation_config!"
            )
        else:
            # Add default value for choose_by if not present
            if ChooseBy.name not in cv_dict.keys():
                cv_map[ChooseBy.name] = ChooseBy(protocol=protocol)

        return cv_map

    @staticmethod
    def _get_hf_dataset_map(
        protocol: Protocol,
        hf_dict: Dict[str, Any]
    ) -> Dict[str, ConfigOption]:
        """
        Create a mapping of HuggingFace dataset options based on the provided configuration.

        Args:
            protocol (Protocol): The selected protocol.
            hf_dict (Dict[str, Any]): The hf_dataset configuration dictionary.

        Returns:
            Dict[str, ConfigOption]: A dictionary mapping hf_dataset option names to their instances.

        Raises:
            ConfigurationException: If an unknown hf_dataset option is encountered or required options are missing.
        """
        hf_map = {}

        for hf_name, hf_value in hf_dict.items():
            try:
                hf_option_class = hf_dataset_dict[hf_name]
                hf_option: ConfigOption = hf_option_class(protocol=protocol, value=hf_value)
                hf_map[hf_name] = hf_option
            except KeyError:
                raise ConfigurationException(
                    f"Unknown hf_dataset option: {hf_name}!"
                )
            except ConfigurationException as e:
                raise ConfigurationException(
                    f"Invalid value for hf_dataset option '{hf_name}': {e}"
                )

        return hf_map

    def _hf_config_updates(self, config_map: Dict[str, ConfigOption]) -> None:
        """
        Apply updates to the configuration map based on the HuggingFace dataset configurations.

        This method ensures that the necessary files (e.g., sequence_file, labels_file, mask_file) are
        created or updated based on the HuggingFace dataset configuration and protocol requirements.

        Args:
            config_map (Dict[str, ConfigOption]): A dictionary mapping configuration option names
                                                  to their instances.

        Raises:
            ConfigurationException: If there are issues with file creation or HuggingFace dataset processing.
        """
        # Ensure the 'hf_dataset' directory exists
        hf_dataset_dir = self._config_file_path / "hf_db"
        hf_dataset_dir.mkdir(exist_ok=True)

        if self._config_dict.get("hf_dataset", None):
            if self.protocol in Protocol.per_sequence_protocols():
                # Update 'sequence_file' in config_map
                sequence_file_path = str(hf_dataset_dir / "sequences.fasta")
                self._update_config_map(config_map, "sequence_file", sequence_file_path)

            elif self.protocol in Protocol.per_residue_protocols():
                # Update 'sequence_file' and 'labels_file' in config_map
                sequence_file_path = str(hf_dataset_dir / "sequences.fasta")
                labels_file_path = str(hf_dataset_dir / "labels.fasta")
                self._update_config_map(config_map, "sequence_file", sequence_file_path)
                self._update_config_map(config_map, "labels_file", labels_file_path)

                # Update 'mask_file' if 'mask_column' is specified
                if self._config_dict["hf_dataset"].get("mask_column", None):
                    mask_file_path = str(hf_dataset_dir / "mask.fasta")
                    self._update_config_map(config_map, "mask_file", mask_file_path)


    @staticmethod
    def _get_config_maps(
        protocol: Protocol,
        config_dict: Dict[str, Any],
        config_file_path: Path = None,
    ) -> Tuple[Dict[str, ConfigOption], Dict[str, ConfigOption], Dict[str, ConfigOption]]:
        """
        Generate configuration, cross-validation, and hf_dataset maps based on the protocol and configuration dictionary.

        Args:
            protocol (Protocol): The selected protocol.
            config_dict (Dict[str, Any]): The configuration dictionary.
            config_file_path (Path, optional): Path to the configuration file directory. Defaults to None.

        Returns:
            Tuple[Dict[str, ConfigOption], Dict[str, ConfigOption], Dict[str, ConfigOption]]:
                - config_map: Mapping of configuration option names to their instances.
                - cv_map: Mapping of cross-validation option names to their instances.
                - hf_map: Mapping of hf_dataset option names to their instances.
        """
        config_map = {}
        cv_map = {}
        hf_map = {}
        contains_cross_validation_config = False

        for config_name in config_dict.keys():
            try:
                if config_name == CROSS_VALIDATION_CONFIG_KEY:
                    cv_map = Configurator._get_cross_validation_map(protocol=protocol,
                                                                    cv_dict=config_dict[config_name])
                    contains_cross_validation_config = True
                elif config_name == HF_DATASET_CONFIG_KEY:
                    hf_map = Configurator._get_hf_dataset_map(protocol=protocol,
                                                              hf_dict=config_dict[config_name])
                else:
                    value = config_dict[config_name]
                    if value == "":  # Ignore empty values
                        continue
                    config_object: ConfigOption = all_options_dict[config_name](protocol=protocol, value=value)
                    config_object.transform_value_if_necessary(config_file_path)
                    config_map[config_name] = config_object
            except KeyError:
                raise ConfigurationException(
                    f"Unknown configuration option: {config_name}!"
                )

        # Add default values for missing configuration options
        all_options_for_protocol: List[ConfigOption] = [
            option for option in all_options_dict.values() if protocol in option.allowed_protocols
        ]
        for option in all_options_for_protocol:
            config_object = option(protocol=protocol)
            if config_object.name not in config_dict.keys() and config_object.default_value != "":
                config_object.transform_value_if_necessary(config_file_path)
                config_map[option.name] = config_object

        # Add default cross-validation method if necessary
        if not contains_cross_validation_config:
            # hold_out by default, does not need any additional parameters
            cv_map[Method.name] = Method(protocol=protocol)
            cv_map[ChooseBy.name] = ChooseBy(protocol=protocol)

        return config_map, cv_map, hf_map

    def _update_config_map(
        self,
        config_map: Dict[str, ConfigOption],
        option_name: str,
        value: Any
    ) -> None:
        """
        Updates an existing ConfigOption in the configuration map or creates a new one if it does not exist.

        This method ensures that the specified configuration option is present in the `config_map`.
        If the `option_name` already exists in `config_map`, its value is updated and any necessary
        transformations are applied. If the `option_name` does not exist, the method attempts to
        create a new ConfigOption instance using the `all_options_dict`. If the `option_name` is
        unrecognized, a `ConfigurationException` is raised.

        Args:
            config_map (Dict[str, ConfigOption]):
                The configuration map to update, where keys are option names and values are ConfigOption instances.
            option_name (str):
                The name of the configuration option to update or add.
            value (Any):
                The new value to assign to the configuration option.

        Raises:
            ConfigurationException:
                If the `option_name` is not recognized or does not correspond to any known configuration option.

        """
        if option_name in config_map:
            config_option = config_map[option_name]
            config_option.value = value
            config_option.transform_value_if_necessary(self._config_file_path)
        else:
            option_class = all_options_dict.get(option_name)
            if option_class:
                config_option = option_class(protocol=self.protocol, value=value)
                config_option.transform_value_if_necessary(self._config_file_path)
                config_map[option_name] = config_option
            else:
                raise ConfigurationException(f"Unknown configuration option: {option_name}")

    @staticmethod
    def _create_hf_files(
        protocol: Protocol,
        config_map: Dict[str, ConfigOption],
        hf_map: Dict[str, ConfigOption]
    ) -> None:
        """
        Creates sequences and, if needed, labels and masks FASTA files based on the HuggingFace
        dataset configuration and protocol requirements. This method downloads and processes the
        HuggingFace dataset according to the selected protocol.

        Args:
            protocol (Protocol): The selected protocol determining how the dataset should be processed.
            config_map (Dict[str, ConfigOption]): A mapping of configuration option names to their respective ConfigOption instances.
            hf_map (Dict[str, ConfigOption]): A mapping of HuggingFace dataset option names to their respective ConfigOption instances.

        Raises:
            ConfigurationException: If there is an issue during the creation of the required files or processing the dataset.
        """
        try:
            process_hf_dataset_to_fasta(protocol, config_map, hf_map)
        except Exception as e:
            raise ConfigurationException(f"Error in _create_hf_files: {e}")

    @staticmethod
    def _check_rules(
        protocol: Protocol,
        config: Dict[str, ConfigOption],
        rules: List[ConfigRule],
        ignore_file_checks: bool
    ):
        """
        Applies a set of validation rules to the provided configuration.

        This method iterates through each rule in the provided list of rules and applies it to the configuration.
        If any rule fails, a `ConfigurationException` is raised with the corresponding failure reason.

        Args:
            protocol (Protocol):
                The selected protocol that dictates which rules are applicable to the configuration.
            config (Dict[str, ConfigOption]):
                A dictionary where keys are the names of configuration options, and values are their corresponding `ConfigOption` instances.
            rules (List[ConfigRule]):
                A list of validation rule objects that will be applied to the configuration.
            ignore_file_checks (bool):
                If set to `True`, file-related checks (such as existence or correctness of file paths) will be ignored.

        Raises:
            ConfigurationException:
                If any validation rule fails, an exception is raised with the reason why the rule was not met.

        """
        config_objects = list(config.values())

        for rule in rules:
            success, reason = rule.apply(
                protocol=protocol,
                config=config_objects,
                ignore_file_checks=ignore_file_checks
            )
            if not success:
                raise ConfigurationException(reason)

    def _verify_config(
        self,
        protocol: Protocol,
        config_map: Dict[str, ConfigOption],
        ignore_file_checks: bool
    ):
        """
        Verify the provided configuration map against a set of validation rules.

        This method ensures that the configuration options are valid for the specified protocol,
        adhere to the defined rules, and have correct values. It also skips file-related checks if
        `ignore_file_checks` is set to `True`.

        Args:
            protocol (Protocol):
                The protocol to validate the configuration against.
            config_map (Dict[str, ConfigOption]):
                A mapping of configuration option names to their respective `ConfigOption` instances.
            ignore_file_checks (bool):
                If `True`, skips validation of file-related options. Defaults to `False`.

        Raises:
            ConfigurationException:
                If any configuration rule is violated or an option is invalid for the given protocol.
        """

        self._check_rules(
            protocol=protocol,
            config=config_map,
            rules=config_option_rules,
            ignore_file_checks=ignore_file_checks
        )

        # Check protocol compatibility and value validity for each configuration option
        for config_object in list(config_map.values()):
            if ignore_file_checks and isinstance(config_object, FileOption):
                continue
            if protocol not in config_object.allowed_protocols:
                raise ConfigurationException(f"{config_object.name} not allowed for protocol {protocol}!")
            if not config_object.check_value():
                raise ConfigurationException(f"{config_object.value} not valid for option {config_object.name}!")

    def _verify_cv_config(
        self,
        protocol: Protocol,
        config_map: Dict[str, ConfigOption],
        cv_config: Dict[str, ConfigOption],
        ignore_file_checks: bool,
    ) -> None:
        """
        Validates the cross-validation configuration against defined rules and ensures compatibility with the selected protocol.

        Args:
            protocol (Protocol): The selected protocol that determines allowed configurations.
            config_map (Dict[str, ConfigOption]): Mapping of general configuration option names to their instances.
            cv_config (Dict[str, ConfigOption]): Mapping of cross-validation option names to their instances.
            ignore_file_checks (bool): If True, file-related checks are ignored during validation.

        Raises:
            ConfigurationException: If any validation rule is violated, required options are missing,
            or incompatible options are used in the cross-validation configuration.
        """

        if Method.name not in cv_config.keys():
            raise ConfigurationException("Required option method is missing from cross_validation_config!")
        method = cv_config[Method.name]

        self._check_rules(
            protocol=protocol,
            config=cv_config,
            rules=cross_validation_rules,
            ignore_file_checks=ignore_file_checks
        )

        self._check_rules(
            protocol=protocol,
            config={**config_map, **cv_config},
            rules=optimization_rules,
            ignore_file_checks=ignore_file_checks
        )

        cv_objects = list(cv_config.values())

        # Ensure that the cross-validation method is compatible with other options
        if method == "hold_out" and len(cv_objects) > 1:
            raise ConfigurationException("Cross-validation method hold_out does not allow any other options!")

        for cv_object in cv_objects:
            if cv_object.cv_methods and method.value not in cv_object.cv_methods:
                raise ConfigurationException(
                    f"Option {cv_object.name} not allowed for cross-validation method {method.name}!"
                )

            if not cv_object.check_value():
                raise ConfigurationException(
                    f"{cv_object.value} not valid for cross-validation option {cv_object.name}!"
                )

    def get_verified_config(self, ignore_file_checks: bool = False) -> Dict[str, Any]:
        """
        Reads the YAML configuration, performs value transformations (such as downloading files),
        and verifies the configuration's correctness.

        Args:
            ignore_file_checks (bool, optional): If True, file-related checks are not performed. Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary with configuration option names as keys and their respective (transformed) values.

        Raises:
            ConfigurationException: If any validation rule is violated or if required options are missing.
        """
        config_map, cv_map, hf_map = self._get_config_maps(
            protocol=self.protocol,
            config_dict=self._config_dict,
            config_file_path=self._config_file_path,
        )

        if hf_map:
            self._check_rules(
                protocol=self.protocol,
                config={**config_map, **hf_map},
                rules=hf_dataset_rules,
                ignore_file_checks=ignore_file_checks
            )
            self._hf_config_updates(config_map)
            self._create_hf_files(
                protocol=self.protocol,
                config_map=config_map,
                hf_map=hf_map,
            )

        else:
            self._check_rules(
                protocol=self.protocol,
                config=config_map,
                rules=local_file_protocol_rules,
                ignore_file_checks=ignore_file_checks
            )

        self._verify_config(
            protocol=self.protocol,
            config_map=config_map,
            ignore_file_checks=ignore_file_checks
        )

        self._verify_cv_config(
            protocol=self.protocol,
            config_map=config_map,
            cv_config=cv_map,
            ignore_file_checks=ignore_file_checks,
        )

        # Prepare the final result dictionary
        result = {config_object.name: config_object.value for config_object in config_map.values()}
        result[CROSS_VALIDATION_CONFIG_KEY] = {cv_object.name: cv_object.value for cv_object in cv_map.values()}
        result[HF_DATASET_CONFIG_KEY] = {hf_object.name: hf_object.value for hf_object in hf_map.values()}

        return result
