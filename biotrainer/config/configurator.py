import os

from ruamel import yaml
from pathlib import Path
from ruamel.yaml import YAMLError
from collections import namedtuple
from typing import Union, List, Dict, Any

from ..protocols import Protocol
from .config_option import ConfigurationException, ConfigOption, FileOption
from .config_rules import MutualExclusive, ProtocolRequires
from .input_options import SequenceFile, LabelsFile, input_options
from .training_options import AutoResume, PretrainedModel, training_options
from .embedding_options import EmbedderName, EmbeddingsFile, embedding_options
from .general_options import general_options
from .model_options import model_options

# Optional attribute for ConfigOption!

protocol_rules = [
    ProtocolRequires(protocol=Protocol.per_residue_protocols(), requires=[SequenceFile, LabelsFile]),
    ProtocolRequires(protocol=Protocol.per_sequence_protocols(), requires=[SequenceFile]),
]

config_option_rules = [
    MutualExclusive(exclusive=[AutoResume, PretrainedModel]),
    MutualExclusive(exclusive=[EmbedderName, EmbeddingsFile])
]

all_options_list: List[
    ConfigOption] = general_options + input_options + model_options + training_options + embedding_options

_ConfigMap = namedtuple("ConfigMap", "key value object")


class Configurator:

    def __init__(self, config_dict: Dict, config_file_path: Path = None):
        if not config_file_path:
            config_file_path = Path("")
        self._config_file_path = config_file_path
        self._config_dict = config_dict
        self.protocol = self._get_protocol_from_config_dict(config_dict)

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]):
        return cls(config_dict=config_dict)

    @classmethod
    def from_config_path(cls, config_path: Union[str, Path]):
        return cls(config_dict=cls._read_config_file(config_path),
                   config_file_path=Path(os.path.dirname(os.path.abspath(config_path))))

    @staticmethod
    def get_options_by_protocol(protocol: Protocol):
        result = []
        for option_class in all_options_list:
            option = option_class(protocol)
            if protocol in option.allowed_protocols:
                result.append(option.to_dict())
        return result

    @staticmethod
    def _verify_config(protocol: Protocol, config_maps: Dict[str, _ConfigMap]) -> bool:
        for config in config_maps.values():
            # Check protocol
            if protocol not in config.object.allowed_protocols:
                raise ConfigurationException(f"{config.object.name} not allowed for protocol {protocol}.")
            # Check value
            if not config.object.is_value_valid(config.value):
                raise ConfigurationException(f"{config.value} not valid for option {config.key}.")

        for rule in protocol_rules:
            success, reason = rule.apply(protocol=protocol,
                                         config=list([config.object for config in config_maps.values()]))
            if not success:
                raise ConfigurationException(reason)

        return True

    @staticmethod
    def _read_config_file(config_path: Union[str, Path], preserve_order: bool = True) -> dict:
        """
        Read config from path to file.

        :param config_path: path to .yml config file
        :param preserve_order: Preserve order in file
        :return: Config file parsed as dict
        """
        with open(config_path, "r") as fp:
            try:
                if preserve_order:
                    return yaml.load(fp, Loader=yaml.Loader)
                else:
                    return yaml.safe_load(fp)
            except YAMLError as e:
                raise ConfigurationException(
                    f"Could not parse configuration file at '{config_path}' as yaml. "
                    "Formatting mistake in config file? "
                    "See Error above for details."
                ) from e

    @staticmethod
    def _get_protocol_from_config_dict(config_dict: Dict[str, Any]):
        try:
            protocol = config_dict["protocol"]
            return Protocol[protocol]
        except KeyError:
            raise ConfigurationException(f"No protocol specified in config file!")

    @staticmethod
    def _get_config_maps(protocol: Protocol, config_dict: Dict[str, Any], config_file_path: Path = None) \
            -> Dict[str, _ConfigMap]:
        all_options_dict: Dict[str, ConfigOption] = {option(protocol).name: option(protocol) for option in
                                                     all_options_list}

        config_maps = {}
        for config_name in config_dict.keys():
            try:
                value = config_dict[config_name]
                config_object: ConfigOption = all_options_dict[config_name]
                # Download file if necessary and allowed
                if issubclass(config_object.__class__, FileOption):
                    value = config_object.download_file_if_necessary(value, str(config_file_path))
                # Make file paths absolute
                if config_file_path:
                    if Path in config_object.possible_types:
                        value = str(config_file_path / value)
                config_maps[config_name] = _ConfigMap(key=config_name, value=value, object=config_object)
            except KeyError:
                raise ConfigurationException(f"Unknown configuration option: {config_name}!")
        # Add default values
        all_options_for_protocol: List[ConfigOption] = [option for option in
                                                        all_options_dict.values()
                                                        if protocol in option.allowed_protocols]
        for option in all_options_for_protocol:
            if option.name not in config_dict.keys() and option.default_value != "":
                value = str(config_file_path / option.default_value) \
                    if Path in option.possible_types else option.default_value

                config_maps[option.name] = _ConfigMap(key=option.name, value=value, object=option)

        return config_maps

    def get_verified_config(self) -> Dict[str, Any]:
        config_maps: Dict[str, _ConfigMap] = self._get_config_maps(protocol=self.protocol,
                                                                   config_dict=self._config_dict,
                                                                   config_file_path=self._config_file_path)
        self._verify_config(protocol=self.protocol, config_maps=config_maps)
        result = {}
        for config_map in config_maps.values():
            result[config_map.key] = config_map.value

        return result
