import os
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple
from datasets import load_dataset, concatenate_datasets

from ruamel import yaml
from ruamel.yaml import YAMLError

from .config_option import ConfigurationException, ConfigOption, FileOption, logger
from .config_rules import (
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
from .input_options import SequenceFile, LabelsFile, input_options
from .model_options import model_options
from .training_options import AutoResume, PretrainedModel, training_options
from .hf_dataset_options import (
    hf_dataset_options,
    HF_DATASET_CONFIG_KEY,
)
from ..protocols import Protocol
from ..utilities import hf_to_fasta

# Define protocol-specific rules
protocol_rules = [
    ProtocolRequires(protocol=Protocol.per_residue_protocols(), requires=[SequenceFile, LabelsFile]),
    ProtocolRequires(protocol=Protocol.per_sequence_protocols(), requires=[SequenceFile]),
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
        try:
            protocol = config_dict["protocol"]
            return Protocol[protocol]
        except KeyError:
            raise ConfigurationException("No protocol specified in config file!")
        except KeyError as e:
            raise ConfigurationException(f"Invalid protocol specified: {config_dict.get('protocol')})") from e

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
                raise ConfigurationException(f"Unknown cross-validation option: {cv_name}!")

        if method == "":
            raise ConfigurationException("Required option method is missing from cross_validation_config!")
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
        required_fields = [opt.name for opt in hf_dataset_options if opt.required]

        for field in required_fields:
            if field not in hf_dict:
                raise ConfigurationException(f"hf_dataset requires the '{field}' field to be set.")

        # Initialize each hf_dataset option
        for hf_name, hf_value in hf_dict.items():
            try:
                hf_option_class = hf_dataset_dict[hf_name]
                hf_option: ConfigOption = hf_option_class(protocol=protocol, value=hf_value)
                hf_map[hf_name] = hf_option
            except KeyError:
                raise ConfigurationException(f"Unknown hf_dataset option: {hf_name}!")
            except ConfigurationException as e:
                raise ConfigurationException(f"Invalid value for hf_dataset option '{hf_name}': {e}")

        return hf_map

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
                raise ConfigurationException(f"Unknown configuration option: {config_name}!")

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

    @staticmethod
    def _hf_log_overwrite_warnings(
            protocol: Protocol,
            config_map: Dict[str, ConfigOption]
    ):
        """
        Logs existing sequence and label files based on the protocol.

        Args:
            protocol (Protocol): The selected protocol.
            config_map (Dict[str, ConfigOption]): Mapping of configuration option names to their instances.
        """
        if protocol in Protocol.per_sequence_protocols():
            sequence_file_option = config_map.get("sequence_file")
            if sequence_file_option and os.path.exists(sequence_file_option.value):
                logger.info("sequence_file already exists. Overwriting it.")

        if protocol in Protocol.per_residue_protocols():
            sequence_file_option = config_map.get("sequence_file")
            labels_file_option = config_map.get("labels_file")

            if sequence_file_option and os.path.exists(sequence_file_option.value):
                logger.info("sequence_file already exists. Overwriting it.")
            if labels_file_option and os.path.exists(labels_file_option.value):
                logger.info("labels_file already exists. Overwriting it.")

    def _load_and_split_dataset(self, hf_map: Dict[str, ConfigOption]) -> Tuple[List[str], List[Any], List[str]]:
        """
        Loads the HuggingFace dataset and splits it into sequences, targets, and set_values.

        Args:
            hf_map (Dict[str, ConfigOption]): Mapping of hf_dataset option names to their instances.

        Returns:
            Tuple[List[str], List[Any], List[str]]: Sequences, targets, and corresponding set values.

        Raises:
            ConfigurationException: If required columns are missing or dataset loading fails.
        """
        path = hf_map["path"].value
        sequence_column = hf_map["sequence_column"].value
        target_column = hf_map["target_column"].value
        subset_name = hf_map["subset"].value if "subset" in hf_map else None

        logger.info(f"Loading HuggingFace dataset from path: {path}")

        if not subset_name:
            dataset = load_dataset(path)
        else:
            dataset = load_dataset(path, subset_name)

        # Collect all available splits
        available_splits = list(dataset.keys())
        if not available_splits:
            raise ConfigurationException(f"No splits found in the dataset at path '{path}'.")

        sequences = []
        targets = []
        set_values = []

        if len(available_splits) > 1:
            logger.info(f"Multiple splits found: {available_splits}. Processing each split separately.")
            for split_name in available_splits:
                split_dataset = dataset[split_name]
                if split_dataset is None:
                    logger.warning(f"Split '{split_name}' is None. Skipping.")
                    continue

                # Verify that the required columns exist
                self._verify_columns(split_dataset, sequence_column, target_column)

                # Determine SET value based on split name
                set_name = self._determine_set_name(split_name)

                split_sequences = split_dataset[sequence_column]
                split_targets = split_dataset[target_column]

                sequences.extend(split_sequences)
                targets.extend(split_targets)
                set_values.extend([set_name] * len(split_sequences))
        else:
            logger.info("Single split found. Performing a train/val/test split.")
            single_split = available_splits[0]
            combined_dataset = dataset[single_split]

            # Verify that the required columns exist
            self._verify_columns(combined_dataset, sequence_column, target_column)

            # Extract the sequence and target columns
            all_sequences = combined_dataset[sequence_column]
            all_targets = combined_dataset[target_column]

            # Perform the split: 70% train, 15% val, 15% test
            train_seq, temp_seq, train_tgt, temp_tgt = train_test_split(
                all_sequences, all_targets, test_size=0.30, random_state=42, shuffle=True
            )
            val_seq, test_seq, val_tgt, test_tgt = train_test_split(
                temp_seq, temp_tgt, test_size=0.50, random_state=42, shuffle=True
            )

            # Assign SET values
            sequences.extend(train_seq)
            targets.extend(train_tgt)
            set_values.extend(["TRAIN"] * len(train_seq))

            sequences.extend(val_seq)
            targets.extend(val_tgt)
            set_values.extend(["VAL"] * len(val_seq))

            sequences.extend(test_seq)
            targets.extend(test_tgt)
            set_values.extend(["TEST"] * len(test_seq))

        return sequences, targets, set_values

    @staticmethod
    def _determine_set_name(split_name: str) -> str:
        """
        Determines the set name (TRAIN, VAL, TEST) based on the split name.

        Args:
            split_name (str): The name of the split.

        Returns:
            str: The correct set name.

        Logs a warning and assigns 'TRAIN' if the split name is unrecognized.
        """
        lower_split = split_name.lower()
        if lower_split.startswith("train"):
            return "TRAIN"
        elif lower_split.startswith("val"):
            return "VAL"
        elif lower_split.startswith("test"):
            return "TEST"
        else:
            logger.warning(f"Unrecognized split name '{split_name}'. Assigning to 'TRAIN'.")
            return "TEST"

    @staticmethod
    def _verify_columns(
            dataset,
            sequence_column: str,
            target_column: str
    ):
        """
        Verifies that the required columns exist in the dataset.

        Args:
            dataset: The dataset to verify.
            sequence_column (str): The name of the sequence column.
            target_column (str): The name of the target column.

        Raises:
            ConfigurationException: If any required column is missing.
        """
        if sequence_column not in dataset.column_names:
            raise ConfigurationException(f"Sequence column '{sequence_column}' not found in the dataset.")
        if target_column not in dataset.column_names:
            raise ConfigurationException(f"Target column '{target_column}' not found in the dataset.")

    @staticmethod
    def _process_and_save_fasta(
            protocol: Protocol,
            sequences: List[str],
            targets: List[Any],
            set_values: List[str],
            config_map: Dict[str, ConfigOption]
    ):
        """
        Processes the sequences and targets and saves them into FASTA files.

        Args:
            protocol (Protocol): The selected protocol.
            sequences (List[str]): List of sequences.
            targets (List[Any]): List of target values.
            set_values (List[str]): List of set values corresponding to each sequence.
            config_map (Dict[str, ConfigOption]): Mapping of configuration option names to their instances.
        """
        if protocol in Protocol.per_sequence_protocols():
            hf_to_fasta(
                sequences=sequences,
                targets=targets,
                set_values=set_values,
                sequences_file_name=config_map["sequence_file"].value
            )
        elif protocol in Protocol.per_residue_protocols():
            hf_to_fasta(
                sequences=sequences,
                targets=targets,
                set_values=set_values,
                sequences_file_name=config_map["sequence_file"].value,
                targets_file_name = config_map["labels_file"].value
            )

    def _create_hf_files(
        self,
        protocol: Protocol,
        config_map: Dict[str, ConfigOption],
        hf_map: Dict[str, ConfigOption],
    ):
        """
        Create sequence and target files based on the hf_dataset configuration. If they exist, they will be overwritten.
        Then, downloads and processes the HuggingFace dataset.

        Args:
            protocol (Protocol): The selected protocol.
            config_map (Dict[str, ConfigOption]): Mapping of configuration option names to their instances.
            hf_map (Dict[str, ConfigOption]): Mapping of hf_dataset option names to their instances.

        Raises:
            ConfigurationException: If there is an issue creating the required files or processing the dataset.
        """
        # Resolve file paths relative to config file directory
        config_dir = self._config_file_path

        sequence_file = config_dir / Path(config_map["sequence_file"].value)
        config_map["sequence_file"].value = str(sequence_file)

        # If protocol requires labels_file, resolve its path
        if protocol in Protocol.per_residue_protocols():
            labels_file = config_dir / Path(config_map["labels_file"].value)
            config_map["labels_file"].value = str(labels_file)

        # Prepare and log existing output files
        self._hf_log_overwrite_warnings(protocol, config_map)

        # Load and split the dataset
        sequences, targets, set_values = self._load_and_split_dataset(hf_map)

        # Process and save to FASTA files
        self._process_and_save_fasta(protocol, sequences, targets, set_values, config_map)

        logger.info("HuggingFace dataset downloaded and processed successfully.")

    @staticmethod
    def _verify_config(
        protocol: Protocol,
        config_map: Dict[str, ConfigOption],
        ignore_file_checks: bool
    ):
        """
        Verify the configuration map against all defined rules.

        Args:
            protocol (Protocol): The selected protocol.
            config_map (Dict[str, ConfigOption]): Mapping of configuration option names to their instances.
            ignore_file_checks (bool): If True, file-related checks are ignored.

        Raises:
            ConfigurationException: If any validation rule is violated.
        """
        config_objects = list(config_map.values())

        # Combine all applicable rules
        all_rules = protocol_rules + config_option_rules
        for rule in all_rules:
            success, reason = rule.apply(
                protocol=protocol,
                config=config_objects,
                ignore_file_checks=ignore_file_checks
            )
            if not success:
                raise ConfigurationException(reason)

        # Check protocol compatibility and value validity for each configuration option
        for config_object in config_objects:
            if ignore_file_checks and isinstance(config_object, FileOption):
                continue
            if protocol not in config_object.allowed_protocols:
                raise ConfigurationException(f"{config_object.name} not allowed for protocol {protocol}!")

            if not config_object.check_value():
                raise ConfigurationException(f"{config_object.value} not valid for option {config_object.name}!")

    @staticmethod
    def _verify_cv_config(
        protocol: Protocol,
        config_map: Dict[str, ConfigOption],
        cv_config: Dict[str, ConfigOption],
        ignore_file_checks: bool,
    ):
        """
        Verify the cross-validation configuration map against all defined rules.

        Args:
            protocol (Protocol): The selected protocol.
            config_map (Dict[str, ConfigOption]): Mapping of configuration option names to their instances.
            cv_config (Dict[str, ConfigOption]): Mapping of cross-validation option names to their instances.
            ignore_file_checks (bool): If True, file-related checks are ignored.

        Raises:
            ConfigurationException: If any validation rule is violated.
        """
        cv_objects = list(cv_config.values())
        config_objects = list(config_map.values())

        if Method.name not in cv_config.keys():
            raise ConfigurationException("Required option method is missing from cross_validation_config!")
        method = cv_config[Method.name]

        # Apply cross-validation specific rules
        for rule in cross_validation_rules:
            success, reason = rule.apply(
                protocol=protocol,
                config=cv_objects,
                ignore_file_checks=ignore_file_checks
            )
            if not success:
                raise ConfigurationException(reason)
        for rule in optimization_rules:
            success, reason = rule.apply(
                protocol,
                config=cv_objects + config_objects,
                ignore_file_checks=ignore_file_checks
            )
            if not success:
                raise ConfigurationException(reason)

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
            self._create_hf_files(
                protocol=self.protocol,
                config_map=config_map,
                hf_map=hf_map,
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
        result = {}
        for config_object in config_map.values():
            result[config_object.name] = config_object.value
        result[CROSS_VALIDATION_CONFIG_KEY] = {}
        for cv_object in cv_map.values():
            result[CROSS_VALIDATION_CONFIG_KEY][cv_object.name] = cv_object.value
        result[HF_DATASET_CONFIG_KEY] = {}
        for hf_object in hf_map.values():
            result[HF_DATASET_CONFIG_KEY][hf_object.name] = hf_object.value

        return result
