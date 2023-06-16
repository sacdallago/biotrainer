import os
from collections import namedtuple
from typing import Union, List, Dict, Any
from pathlib import Path
from ruamel import yaml
from ruamel.yaml import YAMLError

from ..protocols import Protocol
from .config_option import ConfigurationException, ConfigOption
from .config_rules import MutualExclusive, ProtocolRequires
from .input_options import SequenceFile, LabelsFile, MaskFile, input_options
from .training_options import AutoResume, PretrainedModel, training_options
from .embedding_options import EmbedderName, EmbeddingsFile, embedding_options
from .general_options import general_options
from .model_options import model_options

# Optional attribute for ConfigOption!

protocol_rules = [
    ProtocolRequires(protocol=Protocol.per_residue_protocols(), requires=[SequenceFile, LabelsFile]),
    ProtocolRequires(protocol=Protocol.per_protein_protocols(), requires=[SequenceFile]),
]

config_option_rules = [
    MutualExclusive(exclusive=[AutoResume, PretrainedModel]),
    MutualExclusive(exclusive=[EmbedderName, EmbeddingsFile])
]

all_options_list: List[
    ConfigOption] = general_options + input_options + model_options + training_options + embedding_options

_ConfigMap = namedtuple("ConfigMap", "key value object")


class Configurator:

    def __init__(self, config_dict: Dict, input_file_path: Path = None):
        self.protocol = self._get_protocol_from_config_dict(config_dict)
        self.config_map = self._get_config_map(config_dict, self.protocol, input_file_path)

    def get_options_by_protocol(self):
        result = []
        for option_class in all_options_list:
            option = option_class(self.protocol)
            if self.protocol in option.allowed_protocols:
                result.append(option.to_dict())
        return result

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]):
        return cls(config_dict=config_dict)

    @classmethod
    def from_config_path(cls, config_path: Union[str, Path]):
        return cls(config_dict=cls._read_config_file(config_path),
                   input_file_path=Path(os.path.dirname(os.path.abspath(config_path))))

    @staticmethod
    def _read_config_file(config_path: Union[str, Path], preserve_order: bool = True) -> dict:

        """
        Read config from path to file.

        :param config_path: path to .yml config file
        :param preserve_order:
        :return:
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
    def _get_config_map(config_dict: Dict[str, Any], protocol: Protocol, input_file_path: Path = None) -> List[
        _ConfigMap]:
        all_options_dict = {option(protocol).name: option for option in all_options_list}

        config_map = []
        for key in config_dict.keys():
            try:
                value = config_dict[key]
                config_object = all_options_dict[key](protocol)
                if input_file_path:
                    if config_object.category == "input_option":
                        value = input_file_path / value
                config_map.append(
                    _ConfigMap(key=key, value=value, object=config_object))
            except KeyError:
                raise ConfigurationException(f"Unknown configuration option: {key}!")
        return config_map

    def verify_config(self) -> bool:
        for config in self.config_map:
            if self.protocol not in config.object.allowed_protocols:
                raise ConfigurationException(f"{config.object.name} not allowed for protocol {self.protocol}.")
            if not config.object.is_value_valid(config.value):
                raise ConfigurationException(f"{config.value} not valid for option {config.key}.")

        for rule in protocol_rules:
            success, reason = rule.apply(protocol=self.protocol,
                                         config=list([config.object for config in self.config_map]))
            if not success:
                raise ConfigurationException(reason)

        return True
