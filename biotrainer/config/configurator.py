import os

from ruamel import yaml
from pathlib import Path
from itertools import chain
from ruamel.yaml import YAMLError
from typing import Union, List, Dict, Any, Tuple

from .config_option import ConfigOption, ConfigKey
from .deprecated_config import deprecated_config_keys
from .general_config import general_config
from .input_config import input_config
from .model_config import model_config
from .training_config import training_config
from .embedding_config import embedding_config
from .finetuning_config import finetuning_config
from .hf_dataset_config import hf_dataset_config
from .config_exception import ConfigurationException
from .cross_validation_config import cross_validation_config, get_default_cross_validation_config
from .config_validation import validate_config_rules, validate_config_options
from .config_utils import download_file_from_config, is_url, make_path_absolute_if_necessary

from ..protocols import Protocol
from ..input_files import process_hf_dataset_to_fasta


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
            config_file_path: Path = None,
            allow_downloads: bool = True,
    ):
        """
        Initialize a Configurator instance.

        Args:
            config_dict (Dict): The configuration dictionary parsed from the YAML file.
            config_file_path (Path, optional): Path to the configuration file. Defaults to the current directory.
            allow_downloads (bool, optional): Whether to allow downloading files. Defaults to True.
        """
        if not config_file_path:
            config_file_path = Path("")
        self._config_file_path = config_file_path
        self._config_dict = config_dict
        self.allow_downloads = allow_downloads
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
            sub_configs_to_include: List[ConfigKey] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns all possible configuration options as dictionaries for the given protocol.

        Args:
            protocol (Protocol): The protocol to get all options for.
            sub_configs_to_include (List[ConfigKey]): List of sub-configuration keys to include.
        Returns:
            List of config options as dictionaries.
        """
        result = []
        main_config, sub_configs = Configurator._get_relevant_config_options(protocol=protocol,
                                                                             config_keys=[str(config_key.value) for
                                                                                          config_key in
                                                                                          sub_configs_to_include])
        all_config_options = {**main_config,
                              **{k: v for sub_config in sub_configs.values() for k, v in sub_config.items()}}
        for option in all_config_options.values():
            if option.constraints and protocol not in option.constraints.allowed_protocols:
                continue
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
    def _get_relevant_config_options(protocol: Protocol,
                                     config_keys: List[str]) -> (
            Tuple)[Dict[str, ConfigOption], Dict[ConfigKey, Dict[str, ConfigOption]]]:
        main_config = {config_option.name: config_option for config_option in chain(
            general_config(protocol)[1],
            input_config(protocol)[1],
            model_config(protocol)[1],
            training_config(protocol)[1],
            embedding_config(protocol)[1],
        )}
        sub_configs = {config_key: {config_option.name: config_option for config_option in config_options} for
                       config_key, config_options in
                       [hf_dataset_config(protocol), cross_validation_config(protocol), finetuning_config(protocol)]
                       if config_key.value in config_keys}

        return main_config, sub_configs

    def verify_config(self, ignore_file_checks: bool = False) -> Dict[str, Any]:
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
        if ConfigKey.CROSS_VALIDATION.value not in self._config_dict:
            self._config_dict.update(get_default_cross_validation_config())

        # Check config is deprecated
        deprecated_keys_with_replacement = deprecated_config_keys(config_dict=self._config_dict)
        if len(deprecated_keys_with_replacement) > 0:
            deprecation_information = "\n".join([f'Deprecated Key: {key} - Replacement: {replacement}'
                                                 for key, replacement
                                                 in deprecated_keys_with_replacement.items()])
            raise ConfigurationException(f"Config contains deprecated keys:\n{deprecation_information}")

        if not validate_config_rules(protocol=self.protocol,
                                     ignore_file_checks=ignore_file_checks,
                                     config_dict=self._config_dict):
            raise ConfigurationException(f"Provided config is not valid!")

        verified_config = {}
        main_config, sub_configs = self._get_relevant_config_options(protocol=self.protocol,
                                                                     config_keys=[config_key.value for config_key in
                                                                                  ConfigKey.all_subconfig_keys()
                                                                                  if config_key.value in
                                                                                  self._config_dict])
        verified_main_config = validate_config_options(protocol=self.protocol,
                                                       allow_downloads=self.allow_downloads,
                                                       ignore_file_checks=ignore_file_checks,
                                                       config_options=main_config,
                                                       config_dict=self._config_dict,
                                                       config_key=ConfigKey.ROOT)
        verified_config.update(verified_main_config)

        for config_key, config_options in sub_configs.items():
            verified_sub_config = validate_config_options(protocol=self.protocol,
                                                          allow_downloads=self.allow_downloads,
                                                          ignore_file_checks=ignore_file_checks,
                                                          config_options=config_options,
                                                          config_dict=self._config_dict[config_key.value],
                                                          config_key=config_key)
            verified_config[config_key.value] = verified_sub_config
        return verified_config

    def postprocess_config(self, verified_config: Dict[str, Any]) -> Dict[str, Any]:
        main_config, sub_configs = self._get_relevant_config_options(protocol=self.protocol,
                                                                     config_keys=[config_key.value for config_key in
                                                                                  ConfigKey.all_subconfig_keys() if
                                                                                  config_key.value in verified_config])
        all_config_options = {**main_config,
                              **{k: v for sub_config in sub_configs.values() for k, v in sub_config.items()}}

        # Output dir
        if "output_dir" in verified_config:
            output_dir = self._config_file_path / Path(verified_config["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            verified_config["output_dir"] = make_path_absolute_if_necessary(value=verified_config["output_dir"],
                                                                            config_file_path=self._config_file_path)
        else:
            raise ConfigurationException("Verified config is missing output_dir option!")
        output_dir = Path(verified_config["output_dir"])

        if ConfigKey.HF_DATASET.value in verified_config:
            try:
                hf_dataset_dir = output_dir / "hf_db"
                hf_dataset_dir.mkdir(exist_ok=True)
                input_file_path = process_hf_dataset_to_fasta(
                    hf_storage_path=hf_dataset_dir,
                    hf_map=verified_config["hf_dataset"])
                if input_file_path:
                    verified_config["input_file"] = str(input_file_path)
            except Exception as e:
                raise ConfigurationException(f"Error while creating huggingface dataset files: {e}")

        postprocessed_config = {}
        for option_key, option_value in verified_config.items():
            try:
                if isinstance(option_value, dict):
                    postprocessed_config[option_key] = option_value
                    continue  # Nothing to postprocess in sub-configs yet

                config_option: ConfigOption = all_config_options[option_key]
            except KeyError:
                raise ConfigurationException(f"Invalid option key '{option_key}' on verified config.\n"
                                             f"Has the configuration already been verified?")
            # TODO [Refactoring] Special case for ONNX as embedder name - might need to replace file
            #  option with function to determine dynamically
            if (config_option.is_file_option or
                    (config_option.name == "embedder_name" and str(option_value).endswith(".onnx"))):
                if is_url(option_value):
                    if not self.allow_downloads:
                        raise ConfigurationException(f"Downloading files is disabled!")

                    option_value = download_file_from_config(option_name=option_key,
                                                             url=option_value,
                                                             config_file_path=self._config_file_path)
                option_value = make_path_absolute_if_necessary(value=option_value,
                                                               config_file_path=self._config_file_path)

            postprocessed_config[option_key] = option_value

        postprocessed_config["protocol"] = self.protocol.name
        return postprocessed_config

    def get_verified_config(self, ignore_file_checks: bool = False) -> Dict[str, Any]:
        """
        Reads the YAML configuration, performs value transformations (such as downloading files),
        and verifies the configuration's correctness.
        Convenience function to perform validation and postprocessing at once for backwards compatibility.

        Args:
            ignore_file_checks (bool, optional): If True, file-related checks are not performed. Defaults to False
        Returns:
            Dict[str, Any]: Dictionary with configuration option names as keys and their respective (transformed) values
        Raises:
            ConfigurationException: If any validation rule is violated or if required options are missing.
        """
        verified_config = self.verify_config(ignore_file_checks=ignore_file_checks)
        return self.postprocess_config(verified_config)
