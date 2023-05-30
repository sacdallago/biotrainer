from typing import Union, List, Dict, Any
from pathlib import Path
from ruamel import yaml
from ruamel.yaml import YAMLError
from ruamel.yaml.comments import CommentedBase

from ..protocols import Protocols
from .config_option import ConfigurationException, ConfigOption
from .config_rules import MutualExclusive, ProtocolRequires, ProtocolProhibits, ConfigRule
from .input_options import SequenceFile, LabelsFile, MaskFile, input_options
from .training_options import AutoResume, PretrainedModel, training_options
from .embedding_options import EmbedderName, EmbeddingsFile, embedding_options
from .general_options import general_options
from .model_options import model_options

# Optional attribute for ConfigOption!

protocol_rules = [
    ProtocolRequires(protocol=Protocols.per_residue_protocols(), requires=[SequenceFile, LabelsFile]),
    ProtocolRequires(protocol=Protocols.per_protein_protocols(), requires=[SequenceFile]),
    ProtocolProhibits(protocol=Protocols.per_protein_protocols(), prohibits=[LabelsFile, MaskFile])
]

config_option_rules = [
    MutualExclusive(exclusive=[AutoResume, PretrainedModel]),
    MutualExclusive(exclusive=[EmbedderName, EmbeddingsFile])
]

all_options_list: List[
    ConfigOption] = general_options + input_options + model_options + training_options + embedding_options


class Configurator:
    def __init__(self, config_dict: Dict):
        self.config_dict = config_dict
        self.protocol = self.get_protocol_from_config_dict(config_dict)

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
        return cls(config_dict=cls._read_config_file(config_path))

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
    def get_protocol_from_config_dict(config_dict: Dict[str, Any]):
        try:
            protocol = config_dict["protocol"]
            return Protocols[protocol]
        except KeyError:
            raise ConfigurationException(f"No protocol specified in config file!")

    @staticmethod
    def parse_config_dict(config_dict: Dict[str, Any], protocol: Protocols):
        all_options_dict = {option(protocol).name: option for option in all_options_list}
        mapping_dict = {}
        for key in config_dict.keys():
            try:
                mapping_dict[key] = all_options_dict[key]
            except KeyError:
                raise ConfigurationException(f"Unknown configuration option: {key}!")
        return mapping_dict

    def apply_rules(self):
        mapping_dict = self.parse_config_dict(config_dict=self.config_dict, protocol=self.protocol)
        config_options = list(mapping_dict.values())
        for rule in protocol_rules:
            if not rule.apply(protocol=self.protocol, config=config_options):
                raise ConfigurationException(f"Incorrect protocol rule")  # TODO