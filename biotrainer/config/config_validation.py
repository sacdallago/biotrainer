from typing import Dict, Any, Optional

from .config_utils import is_url, is_list_option
from .config_option import ConfigOption, ConfigKey
from .config_exception import ConfigurationException

from ..protocols import Protocol


def validate_config_options(protocol: Protocol,
                            allow_downloads: bool,
                            ignore_file_checks: bool,
                            config_options: Dict[str, ConfigOption],
                            config_dict: Dict[str, Any],
                            config_key: ConfigKey = ConfigKey.ROOT
                            ) -> Dict[str, Any]:
    """
    Validate the configuration dictionary against defined options
    """
    validated_config = {}
    for _, option in config_options.items():
        if option.is_file_option and ignore_file_checks:
            continue

        full_option_name = f"{config_key.name}:{option.name}" if config_key != ConfigKey.ROOT else option.name
        # Check if required option is present
        if option.name not in config_dict:
            if option.required:
                raise ConfigurationException(f"Required option {full_option_name} is missing")
            if option.default is None or option.default == "":
                continue

        # Set value to default if not present in config dict and not required and not None
        value = config_dict.get(option.name, option.default)
        if value is None or str(value) == "":
            continue

        # TODO Improve validation for hyperparameter optimization
        value_is_list_option = is_list_option(value)

        # Check URLs
        if (not allow_downloads or ".py" in str(value)) and is_url(str(value)):
            raise ConfigurationException(f"Downloading files is disabled!")

        # Type checking
        if option.type and option.name in config_dict:
            if not isinstance(value, option.type) and not value_is_list_option:
                raise ConfigurationException(f"{full_option_name} must be of type {option.type}")

        # Constraint validation
        if option.constraints:
            constraints = option.constraints

            # Allowed values
            if constraints.allowed_values and value not in constraints.allowed_values and not value_is_list_option:
                raise ConfigurationException(f"{full_option_name} must be one of {constraints.allowed_values}")

            # Protocol constraints
            if constraints.allowed_protocols and protocol not in constraints.allowed_protocols:
                raise ConfigurationException(f"Option {full_option_name} not valid for protocol {protocol}")

            if constraints.allowed_formats and not any(
                    [str(value).endswith(allowed_format) for allowed_format in constraints.allowed_formats]):
                raise ConfigurationException(f"Option {full_option_name} does not have a valid format: {value} "
                                             f"(allowed formats: {constraints.allowed_formats})")
            # Numeric constraints
            if isinstance(value, (int, float)):
                if constraints.lt is not None and value >= constraints.lt:
                    raise ConfigurationException(f"{full_option_name} must be less than {constraints.lt}")
                if constraints.lte is not None and value > constraints.lte:
                    raise ConfigurationException(
                        f"{full_option_name} must be less than or equal to {constraints.lte}")
                if constraints.gt is not None and value <= constraints.gt:
                    raise ConfigurationException(f"{full_option_name} must be greater than {constraints.gt}")
                if constraints.gte is not None and value < constraints.gte:
                    raise ConfigurationException(
                        f"{full_option_name} must be greater than or equal to {constraints.gte}")
            # Custom constraints
            if constraints.custom_validator:
                validation, error = constraints.custom_validator(value)
                if not validation:
                    raise ConfigurationException(
                        f"Validation for {full_option_name} failed for given value {value}, "
                        f"reason: {error}")

        validated_config[option.name] = value
    return validated_config


def validate_config_rules(protocol: Protocol, ignore_file_checks: bool, config_dict: Dict[str, Any]) -> bool:
    # Input files
    if "hf_dataset" in config_dict:
        if any([file_key in config_dict for file_key in ["sequence_file", "labels_file", "mask_file"]]):
            raise ConfigurationException("If you want to use a dataset from HuggingFace, "
                                         "do not provide any other input file "
                                         "(sequence_file/labels_file/mask_file)!")
    else:
        # Protocol Requires
        if protocol in Protocol.per_residue_protocols():
            if "sequence_file" not in config_dict or "labels_file" not in config_dict and not ignore_file_checks:
                raise ConfigurationException(f"sequence_file and labels_file are required "
                                             f"for given protocol {protocol}!")

        if protocol in Protocol.per_sequence_protocols():
            if "sequence_file" not in config_dict and not ignore_file_checks:
                raise ConfigurationException(f"sequence_file is required for given protocol {protocol}!")

    # Mutual Exclusive
    if "auto_resume" in config_dict and "pretrained_model" in config_dict:
        raise ConfigurationException(f"auto_resume and pretrained_model are mutual exclusive.\n"
                                     f"Use auto_resume in case you need to restart your training job multiple times.\n"
                                     f"Use pretrained_model if you want to continue to train a specific model.")

    if "embedder_name" in config_dict and "embeddings_file" in config_dict:
        raise ConfigurationException(f"embedder_name and embeddings_file are mutual exclusive.\n"
                                     f"Please provide either an embedder_name to calculate embeddings from scratch or \n"
                                     f"an embeddings_file to use pre-computed embeddings.")

    if "use_half_precision" in config_dict and "device" in config_dict:
        if config_dict["use_half_precision"] and config_dict["device"] == "cpu":
            raise ConfigurationException(f"use_half_precision mode is not compatible with embedding on the CPU. "
                                         "(See: https://github.com/huggingface/transformers/issues/11546)")

    # Cross Validation
    if "cross_validation_config" in config_dict:
        cv_config = config_dict["cross_validation_config"]
        if cv_config["method"] == "k_fold" and "k" not in cv_config:
            raise ConfigurationException("Cross validation method k_fold needs k to be set!")
        if cv_config["method"] == "k_fold" and "p" in cv_config:
            raise ConfigurationException("Cross validation method k_fold does not allow p to be set!")

        if cv_config["method"] == "leave_p_out" and "p" not in cv_config:
            raise ConfigurationException("Cross validation method leave_p_out needs p to be set!")
        if cv_config["method"] == "leave_p_out" and "k" in cv_config:
            raise ConfigurationException("Cross validation method leave_p_out does not allow k to be set!")

        any_list_options_in_config_dict = any([is_list_option(value) for value in config_dict.values()])
        if "nested" in cv_config:
            if cv_config["nested"] and ("nested_k" not in cv_config or "search_method" not in cv_config):
                raise ConfigurationException(f"Nested cross validation needs nested_k and search_method to be set!")
            if cv_config["nested"] and not any_list_options_in_config_dict:
                raise ConfigurationException(f"Can only perform hyperparameter optimization "
                                             f"if at least one option to optimize is provided!")
        elif any_list_options_in_config_dict:
            raise ConfigurationException(f"Some options were given multiple values, "
                                         f"but hyperparameter optimization is not activated!")

        if "search_method" in cv_config:
            if cv_config["search_method"] == "random_search" and "n_max_evaluations_random" not in cv_config:
                raise ConfigurationException(
                    f"Search method random_search needs n_max_evaluations_random to be set!")

    return True
