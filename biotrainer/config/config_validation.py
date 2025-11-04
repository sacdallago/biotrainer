from typing import Dict, Any

from .config_utils import is_url, is_list_option
from .config_option import ConfigOption, ConfigKey
from .config_exception import ConfigurationException

from ..protocols import Protocol
from ..embedders import get_predefined_embedder_names


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

        # Constraint validation
        if option.constraints:
            constraints = option.constraints

            # Type checking
            if constraints.type and option.name in config_dict:
                if not isinstance(value, constraints.type) and not value_is_list_option:
                    raise ConfigurationException(f"{full_option_name} must be of type {constraints.type}")

            # Allowed values
            if constraints.allowed_values and value not in constraints.allowed_values and not value_is_list_option:
                raise ConfigurationException(f"{full_option_name} must be one of {constraints.allowed_values}")

            # Protocol constraints
            if constraints.allowed_protocols and protocol not in constraints.allowed_protocols:
                if bool(value):  # Fail if value is not False (this can be ignored)
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

        # Add to config dict if applicable and verified
        if option.add_if and not option.add_if(config_dict):
            continue

        validated_config[option.name] = value
    return validated_config


def validate_config_rules(protocol: Protocol, ignore_file_checks: bool, config_dict: Dict[str, Any]) -> bool:
    # Input files
    if not ignore_file_checks:
        if "hf_dataset" in config_dict:
            if "input_file" in config_dict or "input_data" in config_dict:
                raise ConfigurationException("If you want to use a dataset from HuggingFace, "
                                             "do not provide any other input source!")
        else:
            if "input_file" in config_dict and "input_data" in config_dict:
                raise ConfigurationException("Only provide one of input_file and input_data!")
            if ("input_file" not in config_dict) and ("input_data" not in config_dict):
                raise ConfigurationException("No huggingface dataset or input_file/input_data provided!")

    # Mutual Exclusive
    if "auto_resume" in config_dict and "pretrained_model" in config_dict:
        raise ConfigurationException(f"auto_resume and pretrained_model are mutual exclusive.\n"
                                     f"Use auto_resume in case you need to restart your training job multiple times.\n"
                                     f"Use pretrained_model if you want to continue to train a specific model.")

    if ("auto_resume" in config_dict or "pretrained_model" in config_dict) and "finetuning_config" in config_dict:
        raise ConfigurationException(f"auto_resume and pretrained_model are not supported for finetuning yet!\n")

    if "embedder_name" in config_dict and "custom_tokenizer_config" in config_dict \
            and not str(config_dict["embedder_name"]).endswith(".onnx"):
        raise ConfigurationException("custom_tokenizer_config is only available for onnx embedders at the moment.")

    if "use_half_precision" in config_dict and "device" in config_dict:
        if config_dict["use_half_precision"] and config_dict["device"] == "cpu":
            raise ConfigurationException(f"use_half_precision mode is not compatible with embedding on the CPU. "
                                         "(See: https://github.com/huggingface/transformers/issues/11546)")

    # Other required option must be available
    if "n_reduced_components" in config_dict and not "dimension_reduction_method" in config_dict:
        raise ConfigurationException(f"n_reduced_components requires dimension_reduction_method to be set!")
    if protocol in Protocol.using_per_residue_embeddings() and "dimension_reduction_method" in config_dict:
        raise ConfigurationException(f"dimension_reduction_method is only supported for per-sequence embeddings!")

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

        if (cv_config["method"] == "hold_out" or cv_config["method"] == "leave_p_out") and len(cv_config) > 2:
            raise ConfigurationException("Cross validation config contains unnecessary values!")

        any_list_options_in_config_dict = any([is_list_option(value) for key, value in config_dict.items()
                                               if key != "input_data"])
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

        if cv_config["method"] != "hold_out" and "pretrained_model" in config_dict:
            raise ConfigurationException(f"Using pretrained_model is only possible for hold_out cross validation!")

    # Finetuning
    if "finetuning_config" in config_dict:
        embedder_name = config_dict["embedder_name"]
        if ".onnx" in embedder_name:
            raise ConfigurationException(f"Finetuning an onnx model is not supported!")

        if embedder_name in get_predefined_embedder_names():
            raise ConfigurationException(f"Finetuning {embedder_name} is not supported!")

    return True
